import yaml
from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_agents(llm):
    agents = {}
    agent_configs = load_config("src/inpt_glucose/config/agents.yaml")
    for agent_name, agent_data in agent_configs.items():
        agents[agent_name] = Agent(
            role=agent_data['role'],
            goal=agent_data['goal'],
            backstory=agent_data['backstory'],
            verbose=agent_data.get('verbose', False),
            allow_delegation=agent_data.get('allow_delegation', True),
            llm=llm,
            tools=[DuckDuckGoSearchRun()] if 'DuckDuckGoSearchRun' in agent_data.get('tools', []) else []
        )
    return agents

def create_tasks(agents, progress_note):
    tasks_config = load_config('src/inpt_glucose/config/tasks.yaml')
    tasks = []

    for task_name in sorted(tasks_config.keys()):
        task_info = tasks_config[task_name]
        tasks.append(Task(
            description=f"{task_info['description']}\n\nProgress note: {progress_note}",
            agent=agents[task_info['agent']]
        ))

    return tasks

def run_crew(progress_note, llm):
    agents = create_agents(llm)
    tasks = create_tasks(agents, progress_note)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=True
    )
    return crew.kickoff()