import os

# Set these here or in the ENV when launching the script

#os.environ["SERPER_API_KEY"] = "Serper API Key"  # serper.dev API key
# OpenRouter Configuration, uses OpenAI API
#os.environ["OPENAI_API_KEY"] = "OpenAI API Key"
#os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1" # Or your custom OpenAI API URL
#os.environ["OPENAI_MODEL_NAME"] = "meta-llama/llama-3-8b-instruct" # Model to use

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# import dedent to format the backstory
from textwrap import dedent

search_tool = SerperDevTool()

# Create a travel researcher

researcher = Agent(
  role='Travel Researcher',
  goal='Uncover information about {topic}',
  verbose=True,
  memory=True,
  backstory=dedent("""\
                   You have a long history of uncovering hidden gems and unearthing valuable information
                   that others have overlooked. You are known for your meticulous research skills and
                   ability to connect the dots in ways that others can't. You have a keen eye for detail
                   about {topic} and are always on the lookout for the next big thing.                  
  """),
  tools=[search_tool],
  allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability

itinerary_planner = Agent(
  role='Itinerary Planner',
  goal='Create a fun travel itinerary for {topic}',
  verbose=True,
  memory=True,
  backstory=dedent("""\
                   You excel at planning fun travel itineraries that cater to a wide range of interests.  You have a knack for
                   knowing how many activities are just enough, keeping the client engaged and entertained without
                   overwhelming them. You are known for your ability to create unique and memorable experiences that
                   leave a lasting impression.
  """),
  tools=[search_tool],
  allow_delegation=False
)

# Research task
research_task = Task(
  description=dedent("""\
    Identify the next big trend in {topic}.
    Focus on identifying opportunities that aren't run-of-the-mill, but are manageable.
  """),
  expected_output='A comprehensive list of the best vacation ideas for {topic}.',
  tools=[search_tool],
  agent=researcher,
)

# Choosing a destination
destination_task = Task(
  description=dedent("""\
    Choose the best destination for {topic}.
    Focus on a destination that offers a unique experience and is not overcrowded.
  """),
  expected_output='A single destination that is perfect for {topic}.',
  tools=[search_tool],
  agent=researcher
)

itinerary_task = Task(
  description=dedent("""\
    Create a detailed itinerary for {topic} in the best location suitable.
    Focus on creating a fun and engaging itinerary that caters to a wide range of interests.
  """),
  expected_output='A detailed itinerary for {topic} in the best location suitable.',
  tools=[search_tool],
  agent=itinerary_planner,
)

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[researcher, itinerary_planner],
  tasks=[research_task, destination_task, itinerary_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  cache=True,
  max_rpm=30,
  share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'a family vacation in Europe with tweens'})
print(result)
