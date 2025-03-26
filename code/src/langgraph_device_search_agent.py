import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from copy import deepcopy

# Import required libraries
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFacePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LangGraphDeviceSearchAgent")

# Define the device search state with improved type handling
class DeviceSearchState(BaseModel):
    """State for the device search workflow"""
    # Accept both dict and string types for query
    query: Union[Dict[str, Any], str, None] = Field(default_factory=dict)
    search_criteria: Dict[str, Any] = Field(default_factory=dict)
    found_devices: List[Dict[str, Any]] = Field(default_factory=list)
    upstream_devices: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    downstream_devices: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    affected_services: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    error: Optional[str] = None
    status: str = "initialized"
    rag_context: Optional[str] = None

class LangGraphDeviceSearchAgent:
    """LangGraph-based Device Search Agent integrated with RAG"""

    def __init__(self, llm, rag_system=None):
        """Initialize the agent with LLM and RAG system"""
        self.llm = llm
        self.rag_system = rag_system
        self.graph = self._build_graph()
        logger.info("LangGraph Device Search Agent initialized")

    def _build_graph(self):
        """Build the LangGraph workflow"""
        # Create the graph with the state
        workflow = StateGraph(DeviceSearchState)

        # Add nodes for each step
        workflow.add_node("parse_query", self.parse_query)
        workflow.add_node("search_devices", self.search_devices)
        workflow.add_node("analyze_topology", self.analyze_topology)
        workflow.add_node("format_results", self.format_results)

        # Add edges to define the flow
        workflow.add_edge("parse_query", "search_devices")
        workflow.add_edge("search_devices", "analyze_topology")
        workflow.add_edge("analyze_topology", "format_results")
        workflow.add_edge("format_results", END)

        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "parse_query",
            lambda state: "search_devices" if state.error is None else END
        )

        workflow.add_conditional_edges(
            "search_devices",
            lambda state: "analyze_topology" if state.error is None else END
        )

        # Set the entry point
        workflow.set_entry_point("parse_query")

        # Compile the graph
        return workflow.compile()

    def parse_query(self, state: DeviceSearchState) -> DeviceSearchState:
        """Parse query to extract search criteria with improved string handling"""
        logger.info("Parsing query")
        new_state = deepcopy(state)

        try:
            # Handle different query types
            if isinstance(state.query, str):
                logger.info(f"Converting string query to dict: {state.query}")
                new_state.search_criteria = {"description": state.query}
            elif isinstance(state.query, dict):
                # If already a dict, use it directly
                new_state.search_criteria = state.query
            elif state.query is None:
                # Handle None by creating an empty dict
                new_state.search_criteria = {}
            else:
                # Handle any other type
                new_state.search_criteria = {"raw_value": str(state.query)}

            return new_state

        except Exception as e:
            logger.error(f"Error in parse_query: {e}")
            new_state.error = f"Error parsing query: {str(e)}"
            return new_state

    def search_devices(self, state: DeviceSearchState) -> DeviceSearchState:
        """Search for devices matching criteria with case-insensitive field handling"""
        logger.info("Searching devices")
        new_state = deepcopy(state)

        try:
            # Extract query text for RAG and tracking
            query_text = ""
            if isinstance(state.query, str):
                query_text = state.query
            elif isinstance(state.query, dict) and "description" in state.query:
                query_text = state.query["description"]
            else:
                query_text = json.dumps(state.search_criteria)

            # Get RAG context if available
            context = "Use your knowledge of network topologies."
            if self.rag_system:
                try:
                    # Query the RAG system
                    rag_result = self.rag_system.query("device_search", query_text)
                    if rag_result and "response" in rag_result:
                        # Make the RAG response safe for string formatting
                        rag_response = rag_result['response']
                        safe_response = rag_response.replace("{", "{{").replace("}", "}}")

                        context = f"Use this information from our knowledge base:\n{safe_response}"
                        new_state.rag_context = context

                        logger.info(f"Retrieved {len(rag_result.get('sources', []))} knowledge base entries")
                except Exception as rag_error:
                    logger.warning(f"Error using RAG system: {rag_error}")

            # Use RAG and LLM to simulate device search
            prompt = ChatPromptTemplate.from_template("""<|im_start|>system
You are a network topology expert who can search and find devices in a network.
<|im_end|>
<|im_start|>user
Search for network devices matching these criteria:
{criteria}

{context}

Generate a list of found devices with these EXACT properties (use exactly these field names, all lowercase):
- ci_id: Device identifier (use format like "R001" for routers, "S001" for switches, "FW001" for firewalls)
- name: Descriptive name
- ci_type: Device type (router, switch, firewall, etc.) - MUST BE ALL LOWERCASE
- status: Current status (active, inactive, warning, etc.)
- location: Physical location
- importance: Importance in the network (use values like "high", "medium", "low", "critical")

Respond ONLY with a JSON list of devices that match the criteria. Use EXACTLY the field names shown above.
<|im_end|>
<|im_start|>assistant
""")

            # Format the criteria for the prompt
            criteria_str = json.dumps(state.search_criteria, indent=2)

            # Handle potential format string issues
            safe_criteria_str = criteria_str.replace("{", "{{").replace("}", "}}")

            # Generate device list
            try:
                formatted_prompt = prompt.format(criteria=safe_criteria_str, context=context)
                response = self.llm.invoke(formatted_prompt)
            except ValueError as ve:
                logger.warning(f"Format string error: {ve}, using alternative formatting")
                template = prompt.template
                safe_template = template.replace("{criteria}", criteria_str).replace("{context}",
                                          "Use your knowledge of network devices to find matching devices.")
                response = self.llm.invoke(safe_template)

            # Use the robust JSON extraction and parsing
            devices = self._extract_and_fix_json(response, query_text)

            # Log received devices for debugging
            logger.info(f"Devices before standardization: {devices}")

            # Case-insensitive field standardization
            validated_devices = []
            for device in devices:
                # Create a lowercase keys dictionary for case-insensitive lookup
                lower_device = {k.lower(): v for k, v in device.items()}

                # Now extract fields with fallbacks using the lowercase dictionary
                standardized_device = {
                    "ci_id": None,
                    "name": None,
                    "ci_type": None,
                    "status": None,
                    "location": None,
                    "importance": None
                }

                # ci_id field
                if "ci_id" in lower_device:
                    standardized_device["ci_id"] = lower_device["ci_id"]
                elif "id" in lower_device:
                    standardized_device["ci_id"] = lower_device["id"]
                else:
                    # Determine ID based on type if possible
                    if "firewall" in query_text.lower():
                        standardized_device["ci_id"] = "FW001"
                    elif "router" in query_text.lower():
                        standardized_device["ci_id"] = f"R{len(validated_devices)+1:03d}"
                    elif "switch" in query_text.lower():
                        standardized_device["ci_id"] = f"S{len(validated_devices)+1:03d}"
                    else:
                        standardized_device["ci_id"] = f"DEV{len(validated_devices)+1:03d}"

                # name field
                if "name" in lower_device:
                    standardized_device["name"] = lower_device["name"]
                elif "description" in lower_device:
                    standardized_device["name"] = lower_device["description"]
                else:
                    device_type = "Device"
                    if "ci_type" in lower_device:
                        device_type = lower_device["ci_type"]
                    elif "type" in lower_device:
                        device_type = lower_device["type"]
                    standardized_device["name"] = f"{device_type} {len(validated_devices)+1}"

                # ci_type field - CRITICAL for Gradio display
                if "ci_type" in lower_device:
                    standardized_device["ci_type"] = str(lower_device["ci_type"]).lower()
                elif "type" in lower_device:
                    standardized_device["ci_type"] = str(lower_device["type"]).lower()
                else:
                    # Determine type from ID if possible
                    if standardized_device["ci_id"].startswith("R"):
                        standardized_device["ci_type"] = "router"
                    elif standardized_device["ci_id"].startswith("S"):
                        standardized_device["ci_type"] = "switch"
                    elif standardized_device["ci_id"].startswith("FW"):
                        standardized_device["ci_type"] = "firewall"
                    else:
                        standardized_device["ci_type"] = "unknown"

                # status field
                if "status" in lower_device:
                    standardized_device["status"] = str(lower_device["status"])
                elif "state" in lower_device:
                    standardized_device["status"] = str(lower_device["state"])
                else:
                    standardized_device["status"] = "active"

                # location field
                if "location" in lower_device:
                    standardized_device["location"] = str(lower_device["location"])
                elif "site" in lower_device:
                    standardized_device["location"] = str(lower_device["site"])
                else:
                    standardized_device["location"] = "Unknown"

                # importance field
                if "importance" in lower_device:
                    standardized_device["importance"] = str(lower_device["importance"]).lower()
                elif "criticality" in lower_device:
                    standardized_device["importance"] = str(lower_device["criticality"]).lower()
                elif "priority" in lower_device:
                    standardized_device["importance"] = str(lower_device["priority"]).lower()
                else:
                    standardized_device["importance"] = "medium"

                # Make sure all values are strings
                for key, value in standardized_device.items():
                    if value is None:
                        if key == "ci_id":
                            standardized_device[key] = f"DEV{len(validated_devices)+1:03d}"
                        elif key == "name":
                            standardized_device[key] = "Unknown Device"
                        elif key == "ci_type":
                            standardized_device[key] = "unknown"
                        elif key == "status":
                            standardized_device[key] = "active"
                        elif key == "location":
                            standardized_device[key] = "Unknown"
                        elif key == "importance":
                            standardized_device[key] = "medium"
                    elif not isinstance(value, str):
                        standardized_device[key] = str(value)

                logger.info(f"Standardized device: {standardized_device}")
                validated_devices.append(standardized_device)

            new_state.found_devices = validated_devices
            logger.info(f"Successfully processed {len(validated_devices)} standardized devices")

            return new_state

        except Exception as e:
            logger.error(f"Error in search_devices: {e}")
            new_state.error = f"Error searching devices: {str(e)}"
            return new_state

    def _extract_and_fix_json(self, response, query_text=""):
        """Extract and fix JSON from LLM response with robust error handling"""
        import re
        import json

        # Try multiple approaches to extract valid JSON
        json_str = None

        # First try: Look for JSON code block
        json_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Second try: Look for array pattern
            json_match = re.search(r'(\[\s*\{.*\}\s*\])', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Third try: Scan for first [ to last ]
                start = response.find('[')
                end = response.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_str = response[start:end+1].strip()
                else:
                    # Last resort: Check if the entire response might be JSON with some prefix
                    clean_response = response.strip()
                    start = clean_response.find('[')
                    if start != -1:
                        json_str = clean_response[start:].strip()

        # If we couldn't extract JSON, use fallback
        if not json_str:
            logger.warning("Couldn't extract JSON from response, using fallback")
            return self._get_fallback_devices(query_text)

        logger.info(f"Extracted JSON string: {json_str[:100]}...")

        # Cleanup the JSON string before parsing
        try:
            # Remove control characters
            json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)

            # Fix common JSON formatting issues
            json_str = re.sub(r',\s*\}', '}', json_str)  # Fix trailing commas in objects
            json_str = re.sub(r',\s*\]', ']', json_str)  # Fix trailing commas in arrays
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)  # Quote property names

            # Try to parse the JSON with various fixes
            try:
                parsed_data = json.loads(json_str)

                # Check for field/value format (common in some LLM outputs)
                if isinstance(parsed_data, list) and len(parsed_data) > 0 and "field" in parsed_data[0] and "value" in parsed_data[0]:
                    logger.info("Detected field/value format, converting to standard format")

                    # Convert field/value format to standard device object
                    device_fields = {}
                    for item in parsed_data:
                        field = item.get("field", "").lower()
                        value = item.get("value", "")

                        # Map fields to standard names
                        if field == "description" or field == "name":
                            device_fields["name"] = value
                        elif field == "ci_id" or field == "id" or field == "device id":
                            device_fields["ci_id"] = value
                        elif field == "ci_type" or field == "type" or field == "device type":
                            device_fields["ci_type"] = value
                        elif field == "status":
                            device_fields["status"] = value
                        elif field == "location":
                            device_fields["location"] = value
                        elif field == "importance":
                            device_fields["importance"] = value

                    # Create a proper device object
                    if device_fields:
                        # Create a device using the extracted fields
                        # Use "FW001" as default ci_id if query contains "firewall"
                        default_ci_id = "FW001" if "firewall" in query_text.lower() else "DEV001"

                        device = {
                            "ci_id": device_fields.get("ci_id", default_ci_id),
                            "name": device_fields.get("name", "Unknown Device"),
                            "ci_type": device_fields.get("ci_type", "unknown"),
                            "status": device_fields.get("status", "active"),
                            "location": device_fields.get("location", "Unknown"),
                            "importance": device_fields.get("importance", "medium")
                        }
                        return [device]
                    else:
                        # If we couldn't extract fields, use fallback
                        return self._get_fallback_devices(query_text)

                # Handle other list formats that may not be device objects
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    # Check if these are proper device objects
                    if all("name" in item or "ci_id" in item for item in parsed_data):
                        return parsed_data
                    else:
                        # Not device objects, use fallback
                        logger.warning("Parsed JSON doesn't contain proper device objects")
                        return self._get_fallback_devices(query_text)

                # If not a list, try to make it one
                if not isinstance(parsed_data, list):
                    if isinstance(parsed_data, dict):
                        return [parsed_data]
                    else:
                        return self._get_fallback_devices(query_text)

                return parsed_data

            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed: {e}, attempting fixes")

                # Try with single quote replacement
                try:
                    json_str = json_str.replace("'", '"')
                    parsed_data = json.loads(json_str)
                    return parsed_data
                except json.JSONDecodeError as e2:
                    logger.warning(f"Single quote fix failed: {e2}, attempting line-by-line parsing")

                    # Try to parse the JSON line by line
                    try:
                        device_matches = re.finditer(r'\{[^{}]*\}', json_str)
                        devices = []

                        for match in device_matches:
                            device_str = match.group(0)
                            try:
                                device_str = device_str.replace("'", '"')
                                device_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', device_str)
                                device_obj = json.loads(device_str)
                                devices.append(device_obj)
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid device object: {device_str[:50]}...")

                        if devices:
                            logger.info(f"Successfully parsed {len(devices)} devices from line-by-line approach")
                            return devices
                        else:
                            logger.warning("Line-by-line parsing yielded no valid devices")
                    except Exception as e3:
                        logger.warning(f"Line-by-line parsing failed: {e3}")

                    # Manual extraction with regex as last resort
                    try:
                        ci_ids = re.findall(r'["\']ci_id["\']\s*:\s*["\']([^"\']+)["\']', json_str)
                        names = re.findall(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', json_str)
                        types = re.findall(r'["\'](?:ci_type|type)["\']\s*:\s*["\']([^"\']+)["\']', json_str)

                        if ci_ids or names:
                            devices = []
                            # Use the longest list as our base
                            max_length = max(len(ci_ids), len(names), len(types))

                            for i in range(max_length):
                                device = {
                                    "ci_id": ci_ids[i] if i < len(ci_ids) else f"DEV{i+1:03d}",
                                    "name": names[i] if i < len(names) else f"Device {i+1}",
                                    "ci_type": types[i] if i < len(types) else "unknown",
                                    "status": "active",
                                    "importance": "medium",
                                    "location": "Unknown"
                                }
                                devices.append(device)

                            logger.info(f"Manually constructed {len(devices)} devices from regex extraction")
                            return devices
                    except Exception as e4:
                        logger.warning(f"Manual extraction failed: {e4}")

            # If all parsing attempts failed, use fallback
            logger.warning("All JSON parsing attempts failed, using fallback devices")
            return self._get_fallback_devices(query_text)

        except Exception as e:
            logger.error(f"Error cleaning up JSON: {e}")
            return self._get_fallback_devices(query_text)

    def _get_fallback_devices(self, query_text):
        """Generate fallback devices based on query text"""
        devices = []

        # Check for common device types in the query
        if "router" in query_text.lower():
            devices.append({
                "ci_id": "R001",
                "name": "Core Router 1",
                "ci_type": "router",
                "status": "active",
                "location": "Data Center",
                "importance": "critical"
            })

            if "all" in query_text.lower():
                devices.append({
                    "ci_id": "R002",
                    "name": "Core Router 2",
                    "ci_type": "router",
                    "status": "active",
                    "location": "Backup Data Center",
                    "importance": "critical"
                })

        elif "switch" in query_text.lower():
            devices.append({
                "ci_id": "S001",
                "name": "Distribution Switch 1",
                "ci_type": "switch",
                "status": "active",
                "location": "Main Office",
                "importance": "high"
            })

            if "all" in query_text.lower():
                devices.append({
                    "ci_id": "S002",
                    "name": "Access Switch 1",
                    "ci_type": "switch",
                    "status": "active",
                    "location": "Branch Office",
                    "importance": "medium"
                })

        elif "firewall" in query_text.lower():
            devices.append({
                "ci_id": "FW001",
                "name": "Edge Firewall",
                "ci_type": "firewall",
                "status": "active",
                "location": "Data Center",
                "importance": "critical"
            })

        # If nothing matched or we need more devices, add a generic one
        if not devices or "all" in query_text.lower():
            devices.append({
                "ci_id": "DEV001",
                "name": "Generic Network Device",
                "ci_type": "unknown",
                "status": "active",
                "location": "Network Core",
                "importance": "medium"
            })

        return devices

    def analyze_topology(self, state: DeviceSearchState) -> DeviceSearchState:
        """Analyze upstream and downstream devices"""
        logger.info("Analyzing topology")
        new_state = deepcopy(state)

        try:
            # Process each found device
            for device in new_state.found_devices:
                device_id = device.get("ci_id")

                # Use RAG and LLM to simulate topology analysis
                if device_id:
                    # Generate upstream devices
                    upstream = self._generate_connected_devices(device, "upstream")
                    new_state.upstream_devices[device_id] = upstream

                    # Generate downstream devices
                    downstream = self._generate_connected_devices(device, "downstream")
                    new_state.downstream_devices[device_id] = downstream

                    # Generate affected services
                    services = self._generate_affected_services(device)
                    new_state.affected_services[device_id] = services

            return new_state

        except Exception as e:
            logger.error(f"Error in analyze_topology: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            new_state.error = f"Error analyzing topology: {str(e)}"
            return new_state

    def _generate_connected_devices(self, device, direction):
        """Generate connected devices with consistent field names"""
        # Get standardized device information
        device_type = device.get("ci_type", "").lower()
        device_id = device.get("ci_id", "")
        device_name = device.get("name", "unknown-device")

        result = []
        try:
            # Generate different connections based on device type
            if device_type == "router":
                if direction == "upstream":
                    # Router upstream connections
                    numeric_part = 1
                    if device_id.startswith('R') and device_id[1:].isdigit():
                        numeric_part = int(device_id[1:])

                    upstream_id = f"R{numeric_part - 1:03d}" if numeric_part > 1 else "WAN001"
                    upstream_name = f"core-router-{numeric_part - 1:02d}" if numeric_part > 1 else "wan-edge-01"

                    result = [{
                        "ci_id": upstream_id,
                        "name": upstream_name,
                        "ci_type": "router",
                        "status": "active",
                        "location": "Network Core",
                        "importance": "critical"
                    }]
                else:
                    # Router downstream connections
                    switch_suffix = device_id.replace("R", "").replace("-", "")[:3]
                    if not switch_suffix.isalnum():
                        switch_suffix = "001"

                    result = [
                        {
                            "ci_id": f"S{switch_suffix}A",
                            "name": f"distribution-switch-{device_name[:3]}a",
                            "ci_type": "switch",
                            "status": "active",
                            "location": device.get("location", "Unknown"),
                            "importance": "high"
                        },
                        {
                            "ci_id": f"S{switch_suffix}B",
                            "name": f"distribution-switch-{device_name[:3]}b",
                            "ci_type": "switch",
                            "status": "active",
                            "location": device.get("location", "Unknown"),
                            "importance": "high"
                        }
                    ]
            elif device_type == "switch":
                if direction == "upstream":
                    # Switches connect upstream to routers or other switches
                    if "access" in str(device_name).lower():
                        # Create a descriptive ID
                        dist_id = device_id.replace("S", "").split("1")[0]
                        if not dist_id:
                            dist_id = "01"

                        result = [
                            {
                                "ci_id": f"S{dist_id}0",
                                "name": f"distribution-switch-{dist_id}",
                                "ci_type": "switch",
                                "status": "active",
                                "location": device.get("location", "Unknown"),
                                "importance": "high"
                            }
                        ]
                    else:
                        # Create a sensible router ID
                        router_suffix = device_id.replace("S", "").replace("-", "")[:2]
                        if not router_suffix.isalnum():
                            router_suffix = "01"  # Fallback suffix

                        result = [
                            {
                                "ci_id": f"R{router_suffix}",
                                "name": f"core-router-{router_suffix}",
                                "ci_type": "router",
                                "status": "active",
                                "location": device.get("location", "Unknown"),
                                "importance": "critical"
                            }
                        ]
                else:
                    # Downstream devices
                    device_suffix = device_id.replace("S", "").replace("-", "")[:3]
                    if not device_suffix.isalnum():
                        device_suffix = "001"  # Fallback

                    if "access" in str(device_name).lower():
                        result = [
                            {
                                "ci_id": f"SRV{device_suffix}A",
                                "name": f"server-{device_suffix}-rack-a",
                                "ci_type": "server",
                                "status": "active",
                                "location": device.get("location", "Unknown"),
                                "importance": "high"
                            },
                            {
                                "ci_id": f"SRV{device_suffix}B",
                                "name": f"server-{device_suffix}-rack-b",
                                "ci_type": "server",
                                "status": "active",
                                "location": device.get("location", "Unknown"),
                                "importance": "high"
                            }
                        ]
                    else:
                        # Distribution switches connect to access switches
                        result = [
                            {
                                "ci_id": f"S{device_suffix}1",
                                "name": f"access-switch-{device_suffix}-1",
                                "ci_type": "switch",
                                "status": "active",
                                "location": device.get("location", "Unknown"),
                                "importance": "medium"
                            },
                            {
                                "ci_id": f"S{device_suffix}2",
                                "name": f"access-switch-{device_suffix}-2",
                                "ci_type": "switch",
                                "status": "active",
                                "location": device.get("location", "Unknown"),
                                "importance": "medium"
                            }
                        ]
            elif device_type == "firewall":
                if direction == "upstream":
                    # Firewall connects upstream to router
                    result = [
                        {
                            "ci_id": "R001",
                            "name": "core-router-01",
                            "ci_type": "router",
                            "status": "active",
                            "location": "Network Core",
                            "importance": "critical"
                        }
                    ]
                else:
                  # Firewall connects downstream to DMZ and internal networks
                    result = [
                        {
                            "ci_id": "DMZ001",
                            "name": "dmz-switch-01",
                            "ci_type": "switch",
                            "status": "active",
                            "location": "DMZ",
                            "importance": "high"
                        },
                        {
                            "ci_id": "S001",
                            "name": "internal-switch-01",
                            "ci_type": "switch",
                            "status": "active",
                            "location": "Internal Network",
                            "importance": "high"
                        }
                    ]

            # If we couldn't generate based on type, provide a consistent fallback
            if not result:
                hash_val = abs(hash(str(device_id))) % 1000  # Ensure positive hash value
                result = [
                    {
                        "ci_id": f"DEV{hash_val:03d}",
                        "name": f"connected-device-{direction}",
                        "ci_type": "generic",
                        "status": "active",
                        "location": "Unknown",
                        "importance": "medium"
                    }
                ]

        except Exception as e:
            # Log the error but don't break the workflow
            logger.warning(f"Error generating connected devices for {device_id}: {e}")
            # Return a generic device as fallback
            hash_val = abs(hash(str(device_id))) % 1000  # Ensure positive hash value
            result = [
                {
                    "ci_id": f"DEV{hash_val:03d}",
                    "name": f"connected-device-{direction}",
                    "ci_type": "generic",
                    "status": "active",
                    "location": "Unknown",
                    "importance": "medium"
                }
            ]

        return result

    def _generate_affected_services(self, device):
        """Generate affected services for a device with improved type handling"""
        device_type = device.get("ci_type", "")

        # Fix for the error: Check type before calling .lower()
        importance_value = device.get("importance", "")
        # Convert to string if it's not already
        if not isinstance(importance_value, str):
            importance_value = str(importance_value)
        importance = importance_value.lower()

        # Generate services based on device type and importance
        services = []

        if importance in ["critical", "high"] or "router" in str(device_type).lower():
            services.append({
                "service_id": "SVC001",
                "name": "Customer Portal",
                "status": "active",
                "criticality": "high"
            })

        if "router" in str(device_type).lower() or "firewall" in str(device_type).lower():
            services.append({
                "service_id": "SVC002",
                "name": "VPN Access",
                "status": "active",
                "criticality": "medium"
            })

        if "switch" in str(device_type).lower() and "distribution" in str(device.get("name", "")).lower():
            services.append({
                "service_id": "SVC003",
                "name": "Internal Applications",
                "status": "active",
                "criticality": "medium"
            })

        if "switch" in str(device_type).lower() and "access" in str(device.get("name", "")).lower():
            services.append({
                "service_id": "SVC004",
                "name": "Office Network",
                "status": "active",
                "criticality": "low"
            })

        if "firewall" in str(device_type).lower():
            services.append({
                "service_id": "SVC005",
                "name": "Security Services",
                "status": "active",
                "criticality": "critical"
            })

        return services

    def format_results(self, state: DeviceSearchState) -> DeviceSearchState:
        """Format the results for the response"""
        logger.info("Formatting results")
        return state

    def __call__(self, input_data, mcp_context=None):
        """Process a device search request with improved result handling"""
        logger.info(f"Device search request: {input_data}")

        try:
            # Handle different input formats
            if isinstance(input_data, str):
                # Direct string input
                query = input_data
                logger.info(f"Direct string query: {query}")
            elif isinstance(input_data, dict) and "query" in input_data:
                # Dictionary with query key
                query = input_data.get("query")
                logger.info(f"Query from dict: {query}")
            else:
                # Fallback
                query = input_data
                logger.info(f"Using input directly as query: {type(query)}")

            # Create initial state - DeviceSearchState will handle conversion
            initial_state = DeviceSearchState(query=query)

            # Run the graph
            result = self.graph.invoke(initial_state)
            logger.info(f"Graph execution result type: {type(result)}")

            # Handle different result types
            # Check if result is a dictionary-like object
            if hasattr(result, "get") and callable(result.get):
                # It's a dictionary-like object (AddableValuesDict)
                logger.info("Processing result as dictionary-like object")
                error = result.get("error")
                formatted_result = {
                    "success": error is None,
                    "found_devices": result.get("found_devices", []),
                    "upstream_devices": result.get("upstream_devices", {}),
                    "downstream_devices": result.get("downstream_devices", {}),
                    "affected_services": result.get("affected_services", {}),
                }
                if error:
                    formatted_result["error"] = error
            else:
                # It should be a DeviceSearchState object
                logger.info("Processing result as DeviceSearchState object")
                formatted_result = {
                    "success": not hasattr(result, "error") or result.error is None,
                    "found_devices": getattr(result, "found_devices", []),
                    "upstream_devices": getattr(result, "upstream_devices", {}),
                    "downstream_devices": getattr(result, "downstream_devices", {}),
                    "affected_services": getattr(result, "affected_services", {}),
                }
                if hasattr(result, "error") and result.error:
                    formatted_result["error"] = result.error

            return formatted_result

        except Exception as e:
            logger.error(f"Error in device search agent: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}",
                "found_devices": []
            }
