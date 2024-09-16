import yaml
from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_agents(llm):
    agents_config = load_config('src/inpt_glucose/config/agents.yaml')
    agents = {}

    for agent_name, agent_data in agents_config.items():
        tools = []
        if 'tools' in agent_data:
            for tool in agent_data['tools']:
                if tool == 'DuckDuckGoSearchRun':
                    tools.append(DuckDuckGoSearchRun())
        
        agents[agent_name] = Agent(
            role=agent_data['role'],
            goal=agent_data['goal'],
            backstory=agent_data['backstory'],
            verbose=agent_data['verbose'],
            allow_delegation=agent_data['allow_delegation'],
            llm=llm,
            tools=tools
        )

    return agents

def create_tasks(agents):
    tasks_config = load_config('src/inpt_glucose/config/tasks.yaml')
    tasks = []

    for task_data in tasks_config:
        tasks.append(Task(
            description=task_data['description'],
            agent=agents[task_data['agent']]
        ))

    return tasks

def run_crew(progress_note, llm):
    agents = create_agents(llm)
    tasks = create_tasks(agents)

    glucose_inputs = {'progress note': progress_note}
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        memory=True,
        verbose=True
    )
    return crew.kickoff(inputs=glucose_inputs)