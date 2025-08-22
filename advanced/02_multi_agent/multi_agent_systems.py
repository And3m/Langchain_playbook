#!/usr/bin/env python3
"""
Multi-Agent Systems - Coordinated AI Agents

This example demonstrates how to build and coordinate multiple AI agents
working together to solve complex problems that require specialized skills.

Key concepts:
1. Agent specialization and roles
2. Inter-agent communication
3. Task delegation and coordination
4. Hierarchical agent structures
5. Conflict resolution and consensus
6. Distributed problem solving
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, tool
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage


class AgentRole(Enum):
    """Define different agent roles."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher" 
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: str
    receiver: str
    content: str
    message_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[str]] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def subscribe(self, agent_id: str, message_types: List[str]):
        """Subscribe agent to specific message types."""
        for msg_type in message_types:
            if msg_type not in self.subscribers:
                self.subscribers[msg_type] = []
            if agent_id not in self.subscribers[msg_type]:
                self.subscribers[msg_type].append(agent_id)
    
    def publish(self, message: AgentMessage):
        """Publish message to interested agents."""
        self.messages.append(message)
        self.logger.info(f"Message from {message.sender} to {message.receiver}: {message.message_type}")
        
        # Notify subscribers
        if message.message_type in self.subscribers:
            for subscriber in self.subscribers[message.message_type]:
                if subscriber != message.sender:
                    self.logger.debug(f"Notifying {subscriber} of {message.message_type}")
    
    def get_messages_for(self, agent_id: str, message_type: str = None) -> List[AgentMessage]:
        """Get messages for specific agent."""
        messages = [
            msg for msg in self.messages 
            if msg.receiver == agent_id or msg.receiver == "all"
        ]
        
        if message_type:
            messages = [msg for msg in messages if msg.message_type == message_type]
        
        return messages


class BaseAgent:
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, agent_id: str, role: AgentRole, api_key: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.role = role
        self.message_bus = message_bus
        self.logger = get_logger(f"Agent-{agent_id}")
        self.memory = ConversationBufferMemory()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Role-specific prompt
        self.system_prompt = self._create_system_prompt()
        
        # Task queue
        self.tasks: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = []
    
    def _create_system_prompt(self) -> str:
        """Create role-specific system prompt."""
        base_prompt = f"""You are an AI agent with the role of {self.role.value} in a multi-agent system.
        Your agent ID is {self.agent_id}.
        
        You work collaboratively with other agents to solve complex problems.
        Always be helpful, accurate, and communicate clearly with other agents.
        """
        
        role_specific = {
            AgentRole.COORDINATOR: "You coordinate tasks between agents, delegate work, and ensure project completion.",
            AgentRole.RESEARCHER: "You research information, gather data, and provide factual insights.",
            AgentRole.ANALYST: "You analyze data, identify patterns, and provide analytical insights.",
            AgentRole.WRITER: "You create well-written content, documents, and communications.",
            AgentRole.REVIEWER: "You review work quality, check for errors, and provide feedback.",
            AgentRole.SPECIALIST: "You provide specialized expertise in your domain area."
        }
        
        return base_prompt + "\n" + role_specific.get(self.role, "")
    
    def add_task(self, task: Dict[str, Any]):
        """Add task to agent's queue."""
        task['assigned_time'] = datetime.now()
        task['status'] = 'pending'
        self.tasks.append(task)
        self.logger.info(f"Task added: {task.get('description', 'No description')}")
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task."""
        try:
            task['status'] = 'in_progress'
            task['start_time'] = datetime.now()
            
            # Create prompt for task
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Task: {task['description']}\n\nContext: {task.get('context', '')}")
            ])
            
            # Execute task
            response = self.llm(prompt.format_messages())
            
            # Complete task
            task['status'] = 'completed'
            task['result'] = response.content
            task['completion_time'] = datetime.now()
            
            self.completed_tasks.append(task)
            
            return task
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            self.logger.error(f"Task failed: {e}")
            return task
    
    def send_message(self, receiver: str, content: str, message_type: str, metadata: Dict = None):
        """Send message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=message_type,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.message_bus.publish(message)
    
    def receive_messages(self, message_type: str = None) -> List[AgentMessage]:
        """Receive messages from message bus."""
        return self.message_bus.get_messages_for(self.agent_id, message_type)
    
    def run_cycle(self) -> bool:
        """Run one processing cycle."""
        # Process pending tasks
        if self.tasks:
            task = self.tasks.pop(0)
            self.process_task(task)
            return True
        
        # Check for new messages
        messages = self.receive_messages()
        if messages:
            self._handle_messages(messages)
            return True
        
        return False
    
    def _handle_messages(self, messages: List[AgentMessage]):
        """Handle received messages."""
        for message in messages:
            if message.message_type == "task_assignment":
                # Convert message to task
                task = {
                    'description': message.content,
                    'source': message.sender,
                    'context': message.metadata.get('context', '')
                }
                self.add_task(task)


class CoordinatorAgent(BaseAgent):
    """Coordinator agent that manages other agents."""
    
    def __init__(self, agent_id: str, api_key: str, message_bus: MessageBus):
        super().__init__(agent_id, AgentRole.COORDINATOR, api_key, message_bus)
        self.available_agents: Dict[str, AgentRole] = {}
        self.project_status: Dict[str, Any] = {}
    
    def register_agent(self, agent_id: str, role: AgentRole):
        """Register an agent for coordination."""
        self.available_agents[agent_id] = role
        self.logger.info(f"Registered agent {agent_id} with role {role.value}")
    
    def delegate_task(self, task_description: str, target_role: AgentRole = None, context: str = ""):
        """Delegate task to appropriate agent."""
        # Find suitable agent
        suitable_agents = [
            agent_id for agent_id, role in self.available_agents.items()
            if target_role is None or role == target_role
        ]
        
        if not suitable_agents:
            self.logger.warning(f"No suitable agent found for role {target_role}")
            return False
        
        # Select first available agent (in practice, could be more sophisticated)
        target_agent = suitable_agents[0]
        
        # Send task
        self.send_message(
            receiver=target_agent,
            content=task_description,
            message_type="task_assignment",
            metadata={"context": context}
        )
        
        self.logger.info(f"Delegated task to {target_agent}: {task_description}")
        return True
    
    def coordinate_project(self, project_description: str) -> Dict[str, Any]:
        """Coordinate a complex project across multiple agents."""
        project_id = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Break down project into tasks
        tasks = self._break_down_project(project_description)
        
        # Delegate tasks
        for task in tasks:
            self.delegate_task(
                task['description'],
                task.get('preferred_role'),
                f"Project: {project_description}"
            )
        
        return {
            "project_id": project_id,
            "description": project_description,
            "tasks_delegated": len(tasks),
            "status": "in_progress"
        }
    
    def _break_down_project(self, project_description: str) -> List[Dict[str, Any]]:
        """Break down project into manageable tasks."""
        # In practice, this could use LLM to intelligently break down projects
        if "research" in project_description.lower():
            return [
                {"description": f"Research information about: {project_description}", "preferred_role": AgentRole.RESEARCHER},
                {"description": f"Analyze findings from research", "preferred_role": AgentRole.ANALYST},
                {"description": f"Write summary report", "preferred_role": AgentRole.WRITER},
                {"description": f"Review final output", "preferred_role": AgentRole.REVIEWER}
            ]
        else:
            return [
                {"description": f"Analyze requirements: {project_description}", "preferred_role": AgentRole.ANALYST},
                {"description": f"Create solution proposal", "preferred_role": AgentRole.WRITER}
            ]


class SpecializedAgent(BaseAgent):
    """Specialized agent with domain expertise."""
    
    def __init__(self, agent_id: str, role: AgentRole, api_key: str, message_bus: MessageBus, 
                 specialization: str = ""):
        super().__init__(agent_id, role, api_key, message_bus)
        self.specialization = specialization
        
        # Enhanced system prompt with specialization
        if specialization:
            self.system_prompt += f"\n\nYour specialization is: {specialization}"
    
    def collaborate_with(self, other_agent_id: str, topic: str) -> str:
        """Collaborate with another agent on a topic."""
        # Send collaboration request
        self.send_message(
            receiver=other_agent_id,
            content=f"Let's collaborate on: {topic}",
            message_type="collaboration_request",
            metadata={"topic": topic}
        )
        
        # Wait for response (simplified)
        return f"Collaboration initiated with {other_agent_id} on {topic}"


class MultiAgentSystem:
    """Main system coordinating multiple agents."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize coordinator
        self.coordinator = CoordinatorAgent("coordinator", api_key, self.message_bus)
        self.agents["coordinator"] = self.coordinator
    
    def add_agent(self, agent_id: str, role: AgentRole, specialization: str = "") -> BaseAgent:
        """Add agent to the system."""
        if role == AgentRole.COORDINATOR:
            agent = CoordinatorAgent(agent_id, self.api_key, self.message_bus)
        else:
            agent = SpecializedAgent(agent_id, role, self.api_key, self.message_bus, specialization)
        
        self.agents[agent_id] = agent
        self.coordinator.register_agent(agent_id, role)
        
        self.logger.info(f"Added agent {agent_id} with role {role.value}")
        return agent
    
    def run_simulation(self, project_description: str, cycles: int = 5) -> Dict[str, Any]:
        """Run multi-agent simulation."""
        self.logger.info(f"Starting simulation: {project_description}")
        
        # Start project
        project_info = self.coordinator.coordinate_project(project_description)
        
        # Run simulation cycles
        for cycle in range(cycles):
            self.logger.info(f"Running cycle {cycle + 1}")
            
            # Let each agent process
            for agent in self.agents.values():
                agent.run_cycle()
            
            # Small delay for demonstration
            import time
            time.sleep(0.1)
        
        # Collect results
        results = {
            "project_info": project_info,
            "cycles_run": cycles,
            "agent_results": {}
        }
        
        for agent_id, agent in self.agents.items():
            results["agent_results"][agent_id] = {
                "role": agent.role.value,
                "tasks_completed": len(agent.completed_tasks),
                "tasks_pending": len(agent.tasks)
            }
        
        return results


def demonstrate_multi_agent_system():
    """Demonstrate multi-agent system capabilities."""
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning("⚠️ OpenAI API key not found, showing system structure only")
        
        print("\n" + "="*60)
        print("MULTI-AGENT SYSTEM DEMONSTRATION (STRUCTURE)")
        print("="*60)
        
        # Show system architecture
        print("🏗️ System Architecture:")
        print("   • MessageBus: Central communication hub")
        print("   • CoordinatorAgent: Task delegation and project management")
        print("   • SpecializedAgents: Domain-specific expertise")
        print("   • Agent Roles: Researcher, Analyst, Writer, Reviewer")
        
        print("\n🔄 Communication Flow:")
        print("   1. Coordinator receives project")
        print("   2. Breaks down into specialized tasks")
        print("   3. Delegates to appropriate agents")
        print("   4. Agents collaborate via message bus")
        print("   5. Results aggregated by coordinator")
        
        print("\n💡 Key Features:")
        print("   • Role-based agent specialization")
        print("   • Asynchronous message-based communication")
        print("   • Task delegation and coordination")
        print("   • Collaborative problem solving")
        
        return
    
    logger.info("🤖 Multi-Agent System Demonstration")
    
    # Create multi-agent system
    mas = MultiAgentSystem(api_key)
    
    # Add specialized agents
    mas.add_agent("researcher_1", AgentRole.RESEARCHER, "Technology and AI research")
    mas.add_agent("analyst_1", AgentRole.ANALYST, "Data analysis and insights")
    mas.add_agent("writer_1", AgentRole.WRITER, "Technical writing and documentation")
    mas.add_agent("reviewer_1", AgentRole.REVIEWER, "Quality assurance and editing")
    
    print("\n" + "="*60)
    print("MULTI-AGENT SYSTEM SIMULATION")
    print("="*60)
    
    print(f"🎯 System initialized with {len(mas.agents)} agents:")
    for agent_id, agent in mas.agents.items():
        role_name = agent.role.value
        specialization = getattr(agent, 'specialization', '')
        spec_text = f" (specialized in: {specialization})" if specialization else ""
        print(f"   • {agent_id}: {role_name}{spec_text}")
    
    # Run simulation
    project = "Research and analyze the impact of Large Language Models on software development"
    
    print(f"\n📋 Project: {project}")
    print(f"🚀 Running simulation...")
    
    try:
        results = mas.run_simulation(project, cycles=3)
        
        print("\n📊 Simulation Results:")
        print(f"   Project Status: {results['project_info']['status']}")
        print(f"   Tasks Delegated: {results['project_info']['tasks_delegated']}")
        print(f"   Simulation Cycles: {results['cycles_run']}")
        
        print("\n👥 Agent Activity:")
        for agent_id, stats in results["agent_results"].items():
            print(f"   • {agent_id} ({stats['role']}):")
            print(f"     - Completed tasks: {stats['tasks_completed']}")
            print(f"     - Pending tasks: {stats['tasks_pending']}")
        
        print("\n✅ Multi-agent simulation completed!")
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        print(f"❌ Error: {e}")


def demonstrate_agent_patterns():
    """Demonstrate different multi-agent patterns."""
    logger = get_logger(__name__)
    logger.info("📋 Multi-Agent Patterns")
    
    print("\n" + "="*70)
    print("MULTI-AGENT SYSTEM PATTERNS")
    print("="*70)
    
    patterns = {
        "🏢 Hierarchical Pattern": {
            "Description": "Tree-like structure with clear authority levels",
            "Use Cases": ["Project management", "Military command", "Corporate structure"],
            "Pros": ["Clear authority", "Efficient coordination", "Scalable"],
            "Cons": ["Single point of failure", "Rigid structure", "Bottlenecks"]
        },
        "🌐 Peer-to-Peer Pattern": {
            "Description": "All agents are equal and communicate directly",
            "Use Cases": ["Collaborative research", "Distributed systems", "Democracy"],
            "Pros": ["No single point of failure", "Flexible", "Resilient"],
            "Cons": ["Coordination complexity", "Consensus challenges", "Scalability issues"]
        },
        "🎯 Specialized Team Pattern": {
            "Description": "Agents with different specialized roles work together",
            "Use Cases": ["Software development", "Medical teams", "Research projects"],
            "Pros": ["Domain expertise", "Efficient specialization", "Quality results"],
            "Cons": ["Communication overhead", "Coordination complexity", "Dependencies"]
        },
        "🔄 Pipeline Pattern": {
            "Description": "Agents work in sequence, each adding value",
            "Use Cases": ["Content creation", "Manufacturing", "Data processing"],
            "Pros": ["Clear workflow", "Quality control", "Specialization"],
            "Cons": ["Sequential bottlenecks", "Rigid order", "Error propagation"]
        },
        "🕸️ Network Pattern": {
            "Description": "Complex interconnected agent relationships",
            "Use Cases": ["Social networks", "Market systems", "Ecosystem modeling"],
            "Pros": ["Flexible relationships", "Emergent behavior", "Robust"],
            "Cons": ["Complex dynamics", "Hard to predict", "Control challenges"]
        }
    }
    
    for pattern_name, details in patterns.items():
        print(f"\n{pattern_name}")
        print(f"   Description: {details['Description']}")
        print(f"   Use Cases: {', '.join(details['Use Cases'])}")
        print(f"   ✅ Pros: {', '.join(details['Pros'])}")
        print(f"   ❌ Cons: {', '.join(details['Cons'])}")
    
    print("\n🎯 Design Considerations:")
    considerations = [
        "Communication protocols and message formats",
        "Task decomposition and delegation strategies",
        "Conflict resolution and consensus mechanisms",
        "Load balancing and resource allocation",
        "Fault tolerance and error recovery",
        "Performance monitoring and optimization",
        "Security and access control",
        "Scalability and system growth"
    ]
    
    for consideration in considerations:
        print(f"   • {consideration}")
    
    print("\n⚠️ Common Challenges:")
    challenges = [
        "Communication overhead and latency",
        "Coordination complexity with scale",
        "Conflicting goals and priorities",
        "Resource contention and deadlocks",
        "Maintaining system coherence",
        "Debugging distributed behavior",
        "Version compatibility issues",
        "Performance degradation"
    ]
    
    for challenge in challenges:
        print(f"   ❌ {challenge}")
    
    print("\n✅ Best Practices:")
    best_practices = [
        "Design clear communication protocols",
        "Implement proper error handling and recovery",
        "Use appropriate coordination patterns",
        "Monitor system performance and health",
        "Plan for scalability from the start",
        "Test with realistic scenarios",
        "Document agent behaviors and interactions",
        "Implement security measures"
    ]
    
    for practice in best_practices:
        print(f"   ☑️ {practice}")
    
    print("="*70)


def main():
    """Main function demonstrating multi-agent systems."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Multi-Agent Systems Demonstration")
    
    try:
        demonstrate_multi_agent_system()
        demonstrate_agent_patterns()
        
        print("\n🎯 Multi-Agent Systems Key Takeaways:")
        print("1. Agent specialization improves efficiency and quality")
        print("2. Communication protocols are critical for coordination")
        print("3. Different patterns suit different problem types")
        print("4. Coordination becomes complex with scale")
        print("5. Error handling and fault tolerance are essential")
        print("6. Monitoring and observability enable optimization")
        
        logger.info("✅ Multi-Agent Systems demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        logger.info("💡 Check your API keys and internet connection")


if __name__ == "__main__":
    main()