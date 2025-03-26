import gradio as gr
import json
import logging
import re
from rag_system_updated import NetworkRAGSystem
from full_mcp_implementation import create_mcp_system, AgentContext
from langgraph_device_search_agent import LangGraphDeviceSearchAgent
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_app")

# Initialize the RAG system with correct path
rag_system = NetworkRAGSystem(knowledge_base_path="/content/agentic_rag-mcp_system/network_knowledge_base.txt")

# Initialize the LLM for LangGraph
def init_llm_for_langgraph():
    logger.info("Loading TinyLlama model for LangGraph...")

    try:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )

        # Use max_new_tokens instead of max_length
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # Allow generating up to 512 new tokens
            do_sample=True,      # Enable sampling
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )

        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")

        # Create a simple mock LLM for testing
        from langchain.llms.fake import FakeListLLM
        responses = [
            "This is a mock response for testing purposes. The real LLM could not be initialized."
        ]
        logger.warning("Using mock LLM due to initialization error")
        return FakeListLLM(responses=responses)

# Initialize LangGraph Device Search Agent
llm = init_llm_for_langgraph()
device_search_agent = LangGraphDeviceSearchAgent(llm=llm, rag_system=rag_system)

# Initialize the MCP system with LangGraph agent
mcp_registry = create_mcp_system(rag_system, device_search_agent)

# Create a shared context that persists across interactions
shared_context = AgentContext()

def initialize_system():
    """Initialize the RAG system when the app loads"""
    logger.info("Initializing system...")
    success = rag_system.initialize()
    if success:
        logger.info("System initialized successfully")
    else:
        logger.error("System initialization failed")
    return success

def format_sources(sources):
    """Format source information for display"""
    if not sources:
        return "No sources used"

    return "Sources:\n" + "\n".join([f"- {source}" for source in sources])

def device_search_interface(query):
    """Handle device search queries using MCP with consistent field handling"""
    logger.info(f"Processing device search query: {query}")

    try:
        # Process with MCP system, specifying the agent type explicitly
        result = mcp_registry.process_query(query, agent_type="device_search", context=shared_context)

        # Debug: Log the exact device structure we're receiving
        logger.info(f"Device search result structure: {list(result.keys())}")

        if "error" in result:
            error_msg = result.get("error", "Unknown error")
            return f"## Device Search Results\nThe search could not be completed: {error_msg}\n\nNo sources used."

        # Format the found devices
        output = "## Device Search Results\n\n"

        # Determine which format we're dealing with and extract devices
        devices = []
        if "devices" in result:
            devices = result.get("devices", [])
        elif "found_devices" in result:
            devices = result.get("found_devices", [])

        if devices:
            output += f"### Found {len(devices)} Devices\n\n"
            for device in devices:
                # Normalize keys to lowercase for consistent access
                device_norm = {k.lower(): v for k, v in device.items()}

                # Extract fields with fallbacks
                device_name = device_norm.get('name', device_norm.get('description', 'Unknown Device'))
                device_type = device_norm.get('ci_type', device_norm.get('type', 'Unknown'))
                device_id = device_norm.get('ci_id', device_norm.get('id', 'Unknown'))
                device_status = device_norm.get('status', device_norm.get('state', 'Unknown'))
                device_location = device_norm.get('location', device_norm.get('site', ''))
                device_importance = device_norm.get('importance', device_norm.get('criticality', ''))

                # Format device information
                output += f"- **{device_name}** ({device_type})\n"
                output += f"  - ID: {device_id}\n"
                output += f"  - Status: {device_status}\n"
                if device_location:
                    output += f"  - Location: {device_location}\n"
                if device_importance:
                    output += f"  - Importance: {device_importance}\n"
                output += "\n"
        else:
            output += "No devices found matching your criteria.\n\n"

        # Add topology information from result
        # First check for MCP native format
        if "topology_analysis" in result:
            topology = result.get("topology_analysis", {})
            connections = topology.get("connections", [])
            if connections:
                output += "### Device Connections\n\n"
                for conn in connections:
                    output += f"- {conn.get('from')} → {conn.get('to')} ({conn.get('type', 'connection')})\n"
                output += "\n"

        # Then check for LangGraph format
        upstream_devices = result.get("upstream_devices", {})
        if upstream_devices:
            output += "### Upstream Connections\n\n"
            for device_id, upstream_list in upstream_devices.items():
                if upstream_list:
                    # Find the device name with case-insensitive search
                    device_name = device_id
                    for d in devices:
                        d_lower = {k.lower(): v for k, v in d.items()}
                        if d_lower.get('ci_id', '').lower() == device_id.lower():
                            device_name = d_lower.get('name', device_id)
                            break

                    output += f"**{device_name}** connects to:\n"
                    for upstream in upstream_list:
                        # Case-insensitive lookup for upstream devices
                        u_lower = {k.lower(): v for k, v in upstream.items()}
                        up_name = u_lower.get('name', 'Unknown')
                        up_type = u_lower.get('ci_type', u_lower.get('type', 'Unknown'))
                        output += f"- {up_name} ({up_type})\n"
                    output += "\n"

        downstream_devices = result.get("downstream_devices", {})
        if downstream_devices:
            output += "### Downstream Connections\n\n"
            for device_id, downstream_list in downstream_devices.items():
                if downstream_list:
                    # Find the device name with case-insensitive search
                    device_name = device_id
                    for d in devices:
                        d_lower = {k.lower(): v for k, v in d.items()}
                        if d_lower.get('ci_id', '').lower() == device_id.lower():
                            device_name = d_lower.get('name', device_id)
                            break

                    output += f"**{device_name}** connects to:\n"
                    for downstream in downstream_list:
                        # Case-insensitive lookup for downstream devices
                        d_lower = {k.lower(): v for k, v in downstream.items()}
                        down_name = d_lower.get('name', 'Unknown')
                        down_type = d_lower.get('ci_type', d_lower.get('type', 'Unknown'))
                        output += f"- {down_name} ({down_type})\n"
                    output += "\n"

        # Add service impact information
        affected_services = result.get("affected_services", {})
        if affected_services:
            output += "### Affected Services\n\n"
            all_services = []
            for device_id, service_list in affected_services.items():
                all_services.extend(service_list)

            # Deduplicate services
            unique_services = {}
            for service in all_services:
                service_id = service.get("service_id")
                if service_id and service_id not in unique_services:
                    unique_services[service_id] = service

            for service in unique_services.values():
                # Case-insensitive lookup for services
                s_lower = {k.lower(): v for k, v in service.items()}
                svc_name = s_lower.get('name', 'Unknown')
                svc_crit = s_lower.get('criticality', 'Unknown')
                output += f"- **{svc_name}** (Criticality: {svc_crit})\n"

        # Add sources
        sources = result.get("sources", [])
        if sources:
            output += f"\n## {format_sources(sources)}"

        # Add MCP agent attribution
        output += f"\n\n*Processed by {result.get('agent_type', 'Unknown')} agent ({result.get('agent_id', 'Unknown')})*"

        return output

    except Exception as e:
        logger.error(f"Error in device search interface: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"## Device Search Results\nError processing query: {str(e)}\n\nNo sources used."

def troubleshooting_interface(issue_description, related_cis, logs):
    """Handle troubleshooting queries using MCP"""
    # Combine inputs into a single query
    query = f"Issue: {issue_description}\n"
    if related_cis:
        query += f"Related devices: {related_cis}\n"
    if logs:
        query += f"Logs: {logs}\n"

    # Process with MCP system
    result = mcp_registry.process_query(query, agent_type="troubleshooting", context=shared_context)

    # Format the response
    if "analysis" in result:
        response = result["analysis"]
        suggested_actions = result.get("suggested_actions", [])
        action_text = "\n".join([f"- {action}" for action in suggested_actions]) if suggested_actions else ""

        sources = format_sources(result.get("sources", []))

        # Format output
        output = f"## Analysis\n{response}\n\n"
        if action_text:
            output += f"## Suggested Actions\n{action_text}\n\n"
        output += f"## {sources}"

        # Add MCP agent attribution
        output += f"\n\n*Processed by {result.get('agent_type', 'Unknown')} agent ({result.get('agent_id', 'Unknown')})*"
    else:
        output = f"## Error\n{result.get('error', 'Unknown error occurred')}"

    return output

def knowledge_base_interface(query, doc_type):
    """Handle knowledge base queries using MCP"""
    # Add document type to query if specified
    if doc_type and doc_type != "All Types":
        query = f"{query} (document type: {doc_type})"

    # Process with MCP system
    result = mcp_registry.process_query(query, agent_type="knowledge_base", context=shared_context)

    # Format the response
    if "answer" in result:
        response = result["answer"]
        sources = format_sources(result.get("sources", []))
        related_topics = result.get("related_topics", [])

        # Format output
        output = f"## Knowledge Base Information\n{response}\n\n"

        if related_topics:
            topic_text = ", ".join(related_topics)
            output += f"## Related Topics\n{topic_text}\n\n"

        output += f"## {sources}"

        # Add MCP agent attribution
        output += f"\n\n*Processed by {result.get('agent_type', 'Unknown')} agent ({result.get('agent_id', 'Unknown')})*"
    else:
        output = f"## Error\n{result.get('error', 'Unknown error occurred')}"

    return output

def observability_interface(ci_types, metrics, time_range):
    """Handle observability queries using MCP"""
    # Combine inputs into a single query
    query = f"I need to analyze metrics for CI types: {ci_types}, focusing on these metrics: {metrics}, over time range: {time_range}"

    # Process with MCP system
    result = mcp_registry.process_query(query, agent_type="observability", context=shared_context)

    # Format the response
    if "assessment" in result:
        response = result["assessment"]
        sources = format_sources(result.get("sources", []))

        # Format output
        output = f"## Network Health Assessment\n{response}\n\n"
        output += f"## {sources}"

        # Add MCP agent attribution
        output += f"\n\n*Processed by {result.get('agent_type', 'Unknown')} agent ({result.get('agent_id', 'Unknown')})*"
    else:
        output = f"## Error\n{result.get('error', 'Unknown error occurred')}"

    return output

def incident_resolution_interface(incident_id, title, description, status, priority, affected_cis):
    """Handle incident resolution queries using MCP"""
    # Combine inputs into a single query
    query = f"Incident ID: {incident_id}\nTitle: {title}\nDescription: {description}\nStatus: {status}\nPriority: {priority}\nAffected CIs: {affected_cis}"

    # Process with MCP system
    result = mcp_registry.process_query(query, agent_type="incident_resolution", context=shared_context)

    # Format the response
    if "summary" in result:
        response = result["summary"]
        sources = format_sources(result.get("sources", []))
        action_items = result.get("action_items", [])

        # Format output
        output = f"## Incident Resolution\n{response}\n\n"

        if action_items:
            output += "## Action Items\n"
            for i, item in enumerate(action_items, 1):
                output += f"{i}. {item.get('action')}\n"
                output += f"   - Owner: {item.get('owner')}\n"
                output += f"   - Deadline: {item.get('deadline')}\n\n"

        output += f"## {sources}"

        # Add MCP agent attribution
        output += f"\n\n*Processed by {result.get('agent_type', 'Unknown')} agent ({result.get('agent_id', 'Unknown')})*"
    else:
        output = f"## Error\n{result.get('error', 'Unknown error occurred')}"

    return output

# Create the Gradio app with tabs
with gr.Blocks(title="Network Management with MCP") as demo:
    gr.Markdown("# Network Management with Model Context Protocol (MCP)")
    gr.Markdown("""
    This demo showcases a complete Model Context Protocol (MCP) implementation with RAG-enhanced agents.
    The system maintains context across different agent interactions.
    """)

    # Initialize the system when loading
    system_initialized = gr.Checkbox(value=False, visible=False, label="System Initialized")

    demo.load(initialize_system, None, system_initialized)

    with gr.Tabs():
        with gr.Tab("Device Search"):
            with gr.Group():
                search_query = gr.Textbox(label="Search Query", placeholder="e.g., 'Find all routers in NYC' or 'Show critical devices'")
                search_btn = gr.Button("Search Devices")
                search_output = gr.Markdown(label="Search Results")
                search_btn.click(device_search_interface, [search_query], search_output)

                gr.Markdown("""
                ### Example Queries
                - "Find all active routers"
                - "core devices with high criticality"
                - "Find switches in NYC"
                - "Show all firewall devices"
                - "Find devices that are showing warning status"
                """)

        with gr.Tab("Network Troubleshooting"):
            with gr.Group():
                issue_description = gr.Textbox(label="Issue Description", lines=4, placeholder="Describe the network issue you're experiencing...")
                related_cis = gr.Textbox(label="Related Configuration Items (comma-separated)", placeholder="e.g., router-01, switch-03, firewall-02")
                logs = gr.Textbox(label="Relevant Logs (optional)", lines=4, placeholder="Paste any relevant log entries here...")
                troubleshoot_btn = gr.Button("Analyze Issue")
                troubleshoot_output = gr.Markdown(label="Analysis Results")
                troubleshoot_btn.click(troubleshooting_interface, [issue_description, related_cis, logs], troubleshoot_output)

        with gr.Tab("Network Observability"):
            with gr.Group():
                ci_types = gr.Textbox(label="CI Types to Analyze (comma-separated)", value="router, switch")
                metrics = gr.Textbox(label="Metrics to Analyze (comma-separated)", value="cpu_utilization, latency, memory_utilization")
                time_range = gr.Dropdown(label="Time Range", choices=["last_1h", "last_6h", "last_12h", "last_24h", "last_3d", "last_7d"], value="last_24h")
                observe_btn = gr.Button("Analyze Metrics")
                observe_output = gr.Markdown(label="Analysis Results")
                observe_btn.click(observability_interface, [ci_types, metrics, time_range], observe_output)

        with gr.Tab("Knowledge Base"):
            with gr.Group():
                kb_query = gr.Textbox(label="Search Query", placeholder="e.g., 'Network latency troubleshooting' or 'Firewall best practices'")
                kb_doc_type = gr.Dropdown(label="Document Type", choices=["All Types", "manual", "faq", "best_practice", "troubleshooting_guide", "reference"], value="All Types")
                kb_btn = gr.Button("Search Knowledge Base")
                kb_output = gr.Markdown(label="Search Results")
                kb_btn.click(knowledge_base_interface, [kb_query, kb_doc_type], kb_output)

        with gr.Tab("Incident Resolution"):
            with gr.Group():
                inc_id = gr.Textbox(label="Incident ID", value="INC-001")
                inc_title = gr.Textbox(label="Incident Title", value="Network Outage in NYC Office")
                inc_description = gr.Textbox(label="Description", lines=4, value="Users in the NYC office reported a complete loss of network connectivity at 9:15 AM.")
                inc_status = gr.Dropdown(label="Status", choices=["open", "in_progress", "resolved", "closed"], value="resolved")
                inc_priority = gr.Dropdown(label="Priority", choices=["critical", "high", "medium", "low"], value="critical")
                inc_cis = gr.Textbox(label="Affected CIs (comma-separated)", value="router-nyc-01, switch-nyc-03, firewall-nyc-01")
                inc_btn = gr.Button("Generate Incident Summary")
                inc_output = gr.Markdown(label="Incident Summary")
                inc_btn.click(incident_resolution_interface, [inc_id, inc_title, inc_description, inc_status, inc_priority, inc_cis], inc_output)

        with gr.Tab("About MCP Framework"):
            gr.Markdown("""
            ## Model Context Protocol (MCP) Framework

            This system demonstrates the four key elements of MCP:

            ### 1. Specialized Agents

            Each agent has specific expertise and capabilities:
            - **Troubleshooting Agent**: Diagnoses network issues and suggests resolution steps
            - **Device Search Agent**: Finds devices and analyzes topology relationships
            - **Knowledge Base Agent**: Retrieves information and documentation
            - **Observability Agent**: Analyzes metrics and detects anomalies
            - **Incident Resolution Agent**: Manages incident lifecycle and provides guidance

            ### 2. Context Management

            The MCP framework maintains shared context between agents:
            - Conversation history for continuity between queries
            - Entity memory to track devices, services, and incidents
            - Execution state to share information between agent interactions

            ### 3. Agent Registry

            All agents are registered in a central system:
            - Automatic routing of queries to the most appropriate agent
            - Capability-based discovery for specialized functions
            - Metadata tracking for better transparency

            ### 4. Knowledge Integration

            Relevant knowledge is retrieved dynamically:
            - RAG system integration for retrieving domain knowledge
            - Context-aware queries that incorporate previous interactions
            - Cross-agent knowledge sharing

            ### How It Works

            When you submit a query:
            1. The system analyzes your query to determine the most appropriate agent
            2. The selected agent retrieves relevant knowledge using RAG
            3. Query context is enriched with information from previous interactions
            4. The agent processes the query and builds a structured response
            5. Context is updated for future interactions

            ### Integration with LangGraph

            The Device Search agent leverages LangGraph for a multi-step workflow:
            1. Query parsing: Interprets intent and extracts search criteria
            2. Device search: Finds devices matching criteria
            3. Topology analysis: Maps relationships between devices
            4. Results formatting: Organizes findings in a structured format

            This gives you the benefits of both LangGraph's structured workflow and MCP's context sharing.
            """)

# Entry point to run the app
if __name__ == "__main__":
    # Launch with sharing enabled for Colab access
    demo.launch(share=True)
