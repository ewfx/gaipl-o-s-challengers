import os
import logging
import re
import json
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_system")

class AgentContext:
    """Context container for sharing information between agents"""

    def __init__(self):
        self.conversation_history = []
        self.retrieved_knowledge = {}
        self.entity_memory = {}
        self.execution_state = {}
        self.last_agent = None

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def add_knowledge(self, agent_type: str, knowledge: Dict[str, Any]):
        """Add retrieved knowledge for a specific agent type"""
        self.retrieved_knowledge[agent_type] = knowledge

    def remember_entity(self, entity_type: str, entity_id: str, properties: Dict[str, Any]):
        """Remember an entity like a device or service"""
        if entity_type not in self.entity_memory:
            self.entity_memory[entity_type] = {}
        self.entity_memory[entity_type][entity_id] = properties

    def get_conversation_summary(self, last_n=5):
        """Get a summary of the last n conversation turns"""
        return self.conversation_history[-last_n:] if self.conversation_history else []

    def update_state(self, key: str, value: Any):
        """Update execution state"""
        self.execution_state[key] = value

    def get_state(self, key: str, default=None):
        """Get value from execution state"""
        return self.execution_state.get(key, default)

    def set_last_agent(self, agent_type: str):
        """Set the last agent that processed a request"""
        self.last_agent = agent_type

    def get_last_agent(self):
        """Get the last agent that processed a request"""
        return self.last_agent

class BaseAgent(ABC):
    """Base class for all MCP agents"""

    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.rag_system = None

    def set_rag_system(self, rag_system):
        """Set the RAG system to use for knowledge retrieval"""
        self.rag_system = rag_system

    @abstractmethod
    def process(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Process a query and return a response"""
        pass

    def can_handle(self, capability: str) -> bool:
        """Check if this agent can handle a specific capability"""
        return capability in self.capabilities

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities
        }

class TroubleshootingAgent(BaseAgent):
    """Agent specialized in network troubleshooting"""

    def __init__(self, agent_id: str):
        capabilities = [
            "diagnose_network_issues",
            "interpret_error_messages",
            "suggest_resolution_steps"
        ]
        super().__init__(agent_id, "troubleshooting", capabilities)

    def process(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Process a troubleshooting query"""
        logger.info(f"TroubleshootingAgent processing query: {query}")

        # Use RAG system to retrieve relevant knowledge
        if self.rag_system:
            rag_result = self.rag_system.query("troubleshooting", query)
            context.add_knowledge("troubleshooting", rag_result)

            # Remember any devices mentioned in the query
            if "device" in context.entity_memory:
                for device_id, device in context.entity_memory["device"].items():
                    if device_id.lower() in query.lower() or device.get("name", "").lower() in query.lower():
                        logger.info(f"Found referenced device in query: {device_id}")
                        context.update_state("referenced_device", device_id)

            # Create response
            response = {
                "analysis": rag_result["response"],
                "sources": rag_result["sources"],
                "suggested_actions": self._extract_actions(rag_result["response"])
            }

            # Update context
            context.add_message("agent", json.dumps(response))
            context.set_last_agent("troubleshooting")

            return response
        else:
            return {"error": "RAG system not available"}

    def _extract_actions(self, text: str) -> List[str]:
        """Extract suggested actions from the response text"""
        actions = []

        # Simple extraction - look for numbered lists and bullets
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Match patterns like "1. Check X" or "- Verify Y"
            if (line.startswith('- ') or line.startswith('* ') or
                (len(line) > 2 and line[0].isdigit() and line[1] == '.')):
                action = line[2:].strip() if line[1] in ['.', ' '] else line[1:].strip()
                actions.append(action)

        return actions

class DeviceSearchAgent(BaseAgent):
    """Agent specialized in finding network devices"""

    def __init__(self, agent_id: str, langgraph_agent=None):
        capabilities = [
            "find_devices",
            "analyze_topology",
            "assess_impact"
        ]
        super().__init__(agent_id, "device_search", capabilities)
        # Store reference to LangGraph agent if provided
        self.langgraph_agent = langgraph_agent

    def process(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Process a device search query"""
        logger.info(f"DeviceSearchAgent processing query: {query}")

        # If LangGraph agent is available, use it
        if self.langgraph_agent:
            try:
                logger.info("Using LangGraph agent for device search")
                langgraph_result = self.langgraph_agent(query)

                if not langgraph_result.get("success", False):
                    error_msg = langgraph_result.get("error", "Unknown error")
                    logger.error(f"LangGraph agent error: {error_msg}")
                    return {"error": error_msg}

                # Extract found devices
                found_devices = langgraph_result.get("found_devices", [])

                # Remember devices in context
                for device in found_devices:
                    device_id = device.get("ci_id", "")
                    if device_id:
                        context.remember_entity("device", device_id, {
                            "id": device_id,
                            "name": device.get("name", ""),
                            "type": device.get("ci_type", ""),
                            "status": device.get("status", ""),
                            "location": device.get("location", ""),
                            "importance": device.get("importance", "")
                        })

                # Create response using LangGraph results
                response = {
                    "devices": found_devices,
                    "upstream_devices": langgraph_result.get("upstream_devices", {}),
                    "downstream_devices": langgraph_result.get("downstream_devices", {}),
                    "affected_services": langgraph_result.get("affected_services", {})
                }

                # Update context
                context.add_message("agent", json.dumps(response))
                context.set_last_agent("device_search")

                return response

            except Exception as e:
                logger.error(f"Error using LangGraph agent: {e}")
                # Fall back to RAG-based approach if LangGraph fails

        # Use RAG system as fallback or primary approach
        if self.rag_system:
            # First, get general device knowledge
            rag_result = self.rag_system.query("device_search", query)
            context.add_knowledge("device_search", rag_result)

            # Try to find specific devices in inventory if available
            device_inventory = None
            try:
                inventory_result = self.rag_system.query("device_inventory", query)
                if inventory_result and "retrieved_content" in inventory_result:
                    device_inventory = inventory_result
                    context.add_knowledge("device_inventory", inventory_result)
            except Exception as e:
                logger.warning(f"Error retrieving device inventory: {e}")

            # Extract and remember devices
            devices = self._extract_devices(rag_result["response"], device_inventory)
            for device in devices:
                context.remember_entity("device", device["id"], device)

            # Create response
            response = {
                "devices": devices,
                "topology_analysis": self._analyze_topology(devices, context),
                "sources": rag_result["sources"]
            }

            # Update context
            context.add_message("agent", json.dumps(response))
            context.set_last_agent("device_search")

            return response
        else:
            return {"error": "Neither LangGraph nor RAG system available"}

    def _extract_devices(self, text: str, inventory=None) -> List[Dict[str, Any]]:
        """Extract device information from response text and/or inventory"""
        devices = []

        # First try to extract from inventory if available
        if inventory and "response" in inventory:
            try:
                # Look for patterns like "Device ID: X" or similar structured data
                import re
                device_pattern = r"Device ID: (\w+)[\s\n]+Name: ([^\n]+)[\s\n]+Type: ([^\n]+)[\s\n]+Status: ([^\n]+)"
                matches = re.findall(device_pattern, inventory["response"], re.IGNORECASE)

                for match in matches:
                    device_id, name, device_type, status = match
                    devices.append({
                        "id": device_id.strip(),
                        "name": name.strip(),
                        "type": device_type.strip(),
                        "status": status.strip()
                    })
            except Exception as e:
                logger.warning(f"Error extracting devices from inventory: {e}")

        # If no devices extracted from inventory, try from general response
        if not devices:
            # Simple extraction - look for device mentions
            lines = text.split('\n')
            current_device = {}

            for line in lines:
                line = line.strip()

                # Try to identify device ID or name
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key in ["device", "device id", "id", "name"]:
                        # If we were tracking a device, save it before starting a new one
                        if current_device and "id" in current_device:
                            devices.append(current_device)
                            current_device = {}

                        if key in ["device id", "id"]:
                            current_device["id"] = value
                        else:  # name
                            current_device["name"] = value
                            if "id" not in current_device:
                                # Generate an ID if none exists
                                current_device["id"] = f"DEV{len(devices):03d}"

                    # Capture other device attributes
                    elif key in ["type", "status", "location", "importance"] and current_device:
                        current_device[key] = value

            # Don't forget to add the last device if we were tracking one
            if current_device and "id" in current_device:
                devices.append(current_device)

        return devices

    def _analyze_topology(self, devices: List[Dict[str, Any]], context: AgentContext) -> Dict[str, Any]:
        """Analyze topology relationships between devices"""
        # This would normally draw on knowledge about network topology
        # For this example, we'll use a simplified approach

        topology = {
            "connections": [],
            "dependencies": [],
            "critical_paths": []
        }

        # Identify potential connections based on device types
        for i, device1 in enumerate(devices):
            for j, device2 in enumerate(devices):
                if i == j:
                    continue

                # Simple topology rules
                if device1.get("type", "").lower() == "router" and device2.get("type", "").lower() == "switch":
                    topology["connections"].append({
                        "from": device1["id"],
                        "to": device2["id"],
                        "type": "uplink"
                    })
                    topology["dependencies"].append({
                        "dependent": device2["id"],
                        "depends_on": device1["id"],
                        "reason": "Routing dependency"
                    })

        # Check for critical paths
        for conn in topology["connections"]:
            from_device = next((d for d in devices if d["id"] == conn["from"]), None)
            if from_device and from_device.get("importance", "").lower() in ["critical", "high"]:
                topology["critical_paths"].append({
                    "path_id": f"PATH{len(topology['critical_paths']):03d}",
                    "devices": [conn["from"], conn["to"]],
                    "importance": from_device.get("importance", "unknown")
                })

        return topology

class KnowledgeBaseAgent(BaseAgent):
    """Agent specialized in retrieving knowledge base information"""

    def __init__(self, agent_id: str):
        capabilities = [
            "answer_questions",
            "provide_references",
            "explain_concepts"
        ]
        super().__init__(agent_id, "knowledge_base", capabilities)

    def process(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Process a knowledge base query"""
        logger.info(f"KnowledgeBaseAgent processing query: {query}")

        # Parse document type if specified
        doc_type = None
        if "document type:" in query.lower():
            parts = query.split("document type:", 1)
            query_text = parts[0].strip()
            doc_type = parts[1].strip()
            logger.info(f"Detected document type filter: {doc_type}")
        else:
            query_text = query

        # Enrich query with context from conversation
        enriched_query = self._enrich_query(query_text, context)
        if doc_type:
            enriched_query = f"{enriched_query} (document type: {doc_type})"

        # Use RAG system to retrieve knowledge
        if self.rag_system:
            rag_result = self.rag_system.query("knowledge_base", enriched_query)
            context.add_knowledge("knowledge_base", rag_result)

            # Create response
            response = {
                "answer": rag_result["response"],
                "sources": rag_result["sources"],
                "related_topics": self._extract_related_topics(rag_result["response"], rag_result.get("retrieved_content", []))
            }

            # Update context
            context.add_message("agent", json.dumps(response))
            context.set_last_agent("knowledge_base")

            return response
        else:
            return {"error": "RAG system not available"}

    def _enrich_query(self, query: str, context: AgentContext) -> str:
        """Enrich the query with conversational context"""
        enriched = query

        # Add device context if available
        referenced_device_id = context.get_state("referenced_device")
        if referenced_device_id and "device" in context.entity_memory:
            device = context.entity_memory["device"].get(referenced_device_id)
            if device:
                device_info = f" (regarding {device.get('type', 'device')} {device.get('name', referenced_device_id)})"
                enriched += device_info

        # Add context from last few conversation turns
        last_msgs = context.get_conversation_summary(last_n=2)
        if last_msgs:
            # Extract key information from recent messages
            recent_context = " ".join([msg["content"] for msg in last_msgs if msg["role"] == "user"])
            if recent_context and len(recent_context) > 0:
                enriched = f"{enriched} (context: {recent_context[:100]}...)"

        return enriched

    def _extract_related_topics(self, response: str, content: List[str]) -> List[str]:
        """Extract related topics from the response and retrieved content"""
        topics = set()

        # Look for common technical terms
        tech_terms = [
            "router", "switch", "firewall", "VPN", "ACL", "QoS", "VLAN",
            "routing", "switching", "security", "performance", "latency",
            "DNS", "DHCP", "BGP", "OSPF", "spanning tree", "NAT"
        ]

        # Extract terms from response
        for term in tech_terms:
            if term.lower() in response.lower():
                topics.add(term)

        # Extract from retrieved content
        for doc in content:
            for term in tech_terms:
                if term.lower() in doc.lower() and term not in topics:
                    # Only add a few more to avoid overwhelming
                    if len(topics) < 10:
                        topics.add(term)

        return list(topics)

class ObservabilityAgent(BaseAgent):
    """Agent specialized in network monitoring and observability"""

    def __init__(self, agent_id: str):
        capabilities = [
            "analyze_metrics",
            "detect_anomalies",
            "forecast_trends",
            "health_assessment"
        ]
        super().__init__(agent_id, "observability", capabilities)

    def process(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Process an observability query"""
        logger.info(f"ObservabilityAgent processing query: {query}")

        # Parse metric parameters from query
        ci_types, metrics, time_range = self._parse_metrics_query(query, context)
        logger.info(f"Parsed metrics query: CI Types={ci_types}, Metrics={metrics}, Time Range={time_range}")

        # Use RAG system to retrieve knowledge
        if self.rag_system:
            formatted_query = f"I need to analyze metrics for CI types: {ci_types}, focusing on these metrics: {metrics}, over time range: {time_range}"
            rag_result = self.rag_system.query("observability", formatted_query)
            context.add_knowledge("observability", rag_result)

            # Create response
            response = {
                "assessment": rag_result["response"],
                "sources": rag_result["sources"],
                "parameters": {
                    "ci_types": ci_types,
                    "metrics": metrics,
                    "time_range": time_range
                }
            }

            # Update context - remember analyzed CI types for future reference
            for ci_type in ci_types.split(","):
                ci_type = ci_type.strip()
                if ci_type:
                    context.update_state(f"analyzed_{ci_type}", True)

            context.add_message("agent", json.dumps(response))
            context.set_last_agent("observability")

            return response
        else:
            return {"error": "RAG system not available"}

    def _parse_metrics_query(self, query: str, context: AgentContext) -> tuple:
        """Parse CI types, metrics, and time range from query"""
        # Default values
        default_ci_types = "router, switch"
        default_metrics = "cpu_utilization, latency, memory_utilization"
        default_time_range = "last_24h"

        ci_types = default_ci_types
        metrics = default_metrics
        time_range = default_time_range

        # Try to extract from the query
        query_lower = query.lower()

        # Extract CI types
        if "ci types:" in query_lower or "ci types :" in query_lower:
            parts = re.split(r"ci types\s*:", query_lower, 1)
            if len(parts) > 1:
                ci_part = parts[1].split(",", 1)[0].strip()
                if ci_part:
                    ci_types = ci_part
        elif "devices:" in query_lower or "devices :" in query_lower:
            parts = re.split(r"devices\s*:", query_lower, 1)
            if len(parts) > 1:
                ci_part = parts[1].split(",", 1)[0].strip()
                if ci_part:
                    ci_types = ci_part

        # Extract metrics
        if "metrics:" in query_lower or "metrics :" in query_lower:
            parts = re.split(r"metrics\s*:", query_lower, 1)
            if len(parts) > 1:
                metrics_part = parts[1].split(",", 1)[0].strip()
                if metrics_part:
                    metrics = metrics_part

        # Extract time range
        time_ranges = ["last_1h", "last_6h", "last_12h", "last_24h", "last_3d", "last_7d"]
        for tr in time_ranges:
            if tr in query_lower:
                time_range = tr
                break

        # If devices were previously found in context, use them
        if "device" in context.entity_memory and not "ci types:" in query_lower:
            device_types = set()
            for device_id, device in context.entity_memory["device"].items():
                if "type" in device:
                    device_types.add(device["type"].lower())
            if device_types:
                ci_types = ", ".join(device_types)

        return ci_types, metrics, time_range

class IncidentResolutionAgent(BaseAgent):
    """Agent specialized in incident management"""

    def __init__(self, agent_id: str):
        capabilities = [
            "incident_analysis",
            "resolution_guidance",
            "impact_assessment",
            "root_cause_analysis"
        ]
        super().__init__(agent_id, "incident_resolution", capabilities)

    def process(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Process an incident resolution query"""
        logger.info(f"IncidentResolutionAgent processing query: {query}")

        # Parse incident parameters
        incident_id, title, description, status, priority, affected_cis = self._parse_incident_query(query)

        # Use information from context to enhance query if needed
        if affected_cis == "" and "device" in context.entity_memory:
            # Use devices from memory if no CIs specified
            affected_devices = []
            for device_id, device in context.entity_memory["device"].items():
                affected_devices.append(device.get("name", device_id))
            if affected_devices:
                affected_cis = ", ".join(affected_devices)

        # Use RAG system to retrieve knowledge
        if self.rag_system:
            formatted_query = f"Incident ID: {incident_id}\nTitle: {title}\nDescription: {description}\nStatus: {status}\nPriority: {priority}\nAffected CIs: {affected_cis}"
            rag_result = self.rag_system.query("incident_resolution", formatted_query)
            context.add_knowledge("incident_resolution", rag_result)

            # Create response
            response = {
                "summary": rag_result["response"],
                "sources": rag_result["sources"],
                "incident_details": {
                    "id": incident_id,
                    "title": title,
                    "status": status,
                    "priority": priority,
                    "affected_cis": affected_cis.split(", ") if affected_cis else []
                }
            }

            # Extract action items if any
            action_items = self._extract_action_items(rag_result["response"])
            if action_items:
                response["action_items"] = action_items

            # Update context
            context.add_message("agent", json.dumps(response))
            context.set_last_agent("incident_resolution")

            # Remember this incident in context
            context.remember_entity("incident", incident_id, {
                "id": incident_id,
                "title": title,
                "status": status,
                "priority": priority
            })

            return response
        else:
            return {"error": "RAG system not available"}

    def _parse_incident_query(self, query: str) -> tuple:
        """Parse incident details from query"""
        # Default values
        incident_id = "INC-001"
        title = "Network Issue"
        description = ""
        status = "open"
        priority = "medium"
        affected_cis = ""

        # Try to extract from the query
        lines = query.split('\n')
        for line in lines:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if "id" in key:
                    incident_id = value
                elif "title" in key:
                    title = value
                elif "description" in key:
                    description = value
                elif "status" in key:
                    status = value.lower()
                elif "priority" in key:
                    priority = value.lower()
                elif "affected" in key and "ci" in key:
                    affected_cis = value

        return incident_id, title, description, status, priority, affected_cis

    def _extract_action_items(self, text: str) -> List[Dict[str, str]]:
        """Extract action items from incident resolution text"""
        action_items = []

        # Look for sections that indicate actions
        sections = [
            "next steps", "action items", "recommendations",
            "required actions", "follow-up"
        ]

        lines = text.split('\n')
        in_action_section = False

        for i, line in enumerate(lines):
            line = line.strip().lower()

            # Check if we're entering an action section
            for section in sections:
                if section in line and ":" in line:
                    in_action_section = True
                    break

            # If in action section, look for numbered or bulleted items
            if in_action_section:
                orig_line = lines[i].strip()
                if (orig_line.startswith('- ') or orig_line.startswith('* ') or
                    (len(orig_line) > 2 and orig_line[0].isdigit() and orig_line[1] == '.')):

                    action = orig_line[2:].strip() if orig_line[1] in ['.', ' '] else orig_line[1:].strip()

                    # Try to identify owner and deadline if present
                    owner = "Unassigned"
                    deadline = "Not specified"

                    if "owner:" in action.lower() or "assigned to:" in action.lower():
                        parts = re.split(r"owner:|assigned to:", action.lower(), 1)
                        if len(parts) > 1:
                            potential_owner = parts[1].split(",", 1)[0].strip()
                            if potential_owner:
                                owner = potential_owner
                                action = parts[0].strip()

                    if "by:" in action.lower() or "deadline:" in action.lower() or "due:" in action.lower():
                        deadline_patterns = [r"by:", r"deadline:", r"due:"]
                        for pattern in deadline_patterns:
                            if re.search(pattern, action.lower()):
                                parts = re.split(pattern, action.lower(), 1)
                                if len(parts) > 1:
                                    potential_deadline = parts[1].split(",", 1)[0].strip()
                                    if potential_deadline:
                                        deadline = potential_deadline
                                        action = parts[0].strip()

                    action_items.append({
                        "action": action,
                        "owner": owner,
                        "deadline": deadline
                    })

        return action_items

class MCPRegistry:
    """Central registry for MCP agents"""

    def __init__(self):
        self.agents = {}
        self.rag_system = None
        self.default_context = AgentContext()

    def register_agent(self, agent: BaseAgent):
        """Register an agent in the system"""
        self.agents[agent.agent_id] = agent
        if self.rag_system:
            agent.set_rag_system(self.rag_system)
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")

    def set_rag_system(self, rag_system):
        """Set the RAG system for all agents"""
        self.rag_system = rag_system
        for agent in self.agents.values():
            agent.set_rag_system(rag_system)
        logger.info("RAG system configured for all agents")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)

    def get_agent_by_type(self, agent_type: str) -> Optional[BaseAgent]:
        """Get the first agent of a specific type"""
        for agent in self.agents.values():
            if agent.agent_type == agent_type:
                return agent
        return None

    def find_agent_for_capability(self, capability: str) -> Optional[BaseAgent]:
        """Find an agent that can handle a specific capability"""
        for agent in self.agents.values():
            if agent.can_handle(capability):
                return agent
        return None

    def find_agent_for_query(self, query: str) -> Optional[BaseAgent]:
        """Find the most appropriate agent for a query"""
        # Simple keyword-based routing
        query_lower = query.lower()

        # Define keywords for each agent type
        routing_map = {
            "troubleshooting": ["problem", "issue", "error", "not working", "troubleshoot", "fix", "broken"],
            "device_search": ["find", "search", "device", "router", "switch", "firewall", "topology"],
            "knowledge_base": ["what is", "how to", "explain", "documentation", "best practice"],
            "observability": ["monitor", "metrics", "performance", "trend", "utilization", "health"],
            "incident_resolution": ["incident", "outage", "resolution", "root cause", "impact"]
        }

        # Score each agent type based on keyword matches
        scores = {agent_type: 0 for agent_type in routing_map}
        for agent_type, keywords in routing_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[agent_type] += 1

        # Check for specific formats that clearly indicate agent type
        if "incident id:" in query_lower:
            scores["incident_resolution"] += 10
        elif "ci types:" in query_lower or "metrics:" in query_lower:
            scores["observability"] += 10
        elif "document type:" in query_lower:
            scores["knowledge_base"] += 10

        # Get the agent type with the highest score
        if any(scores.values()):
            best_agent_type = max(scores, key=scores.get)
            logger.info(f"Query routed to {best_agent_type} agent (score: {scores[best_agent_type]})")

            # Find an agent of this type
            for agent in self.agents.values():
                if agent.agent_type == best_agent_type:
                    return agent

        # Fallback to knowledge base agent
        logger.info("No specific agent match, falling back to knowledge base agent")
        for agent in self.agents.values():
            if agent.agent_type == "knowledge_base":
                return agent

        # Last resort: return the first agent
        return next(iter(self.agents.values())) if self.agents else None

    def process_query(self, query: str, agent_type: str = None, context: Optional[AgentContext] = None) -> Dict[str, Any]:
        """Process a query with the most appropriate or specified agent"""
        if context is None:
            context = self.default_context

        # Find the appropriate agent
        agent = None
        if agent_type:
            # If agent type is specified, use it directly
            agent = self.get_agent_by_type(agent_type)

        if not agent:
            # Otherwise find the most appropriate agent
            agent = self.find_agent_for_query(query)

        if not agent:
            return {"error": "No suitable agent found"}

        # Update context with the user query
        context.add_message("user", query)

        # Process the query
        result = agent.process(query, context)

        # Enrich result with agent information
        result["agent_id"] = agent.agent_id
        result["agent_type"] = agent.agent_type

        return result

# Factory function to create a complete MCP system
def create_mcp_system(rag_system, langgraph_agent=None):
    """Create and initialize a complete MCP system"""
    # Create registry
    registry = MCPRegistry()

    # Create all agents
    troubleshooting_agent = TroubleshootingAgent("troubleshoot-1")
    device_search_agent = DeviceSearchAgent("device-search-1", langgraph_agent)
    knowledge_agent = KnowledgeBaseAgent("knowledge-1")
    observability_agent = ObservabilityAgent("observability-1")
    incident_agent = IncidentResolutionAgent("incident-1")

    # Register agents
    registry.register_agent(troubleshooting_agent)
    registry.register_agent(device_search_agent)
    registry.register_agent(knowledge_agent)
    registry.register_agent(observability_agent)
    registry.register_agent(incident_agent)

    # Set RAG system
    registry.set_rag_system(rag_system)

    return registry
