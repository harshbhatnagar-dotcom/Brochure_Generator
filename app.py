from scraper import fetch_website_contents,fetch_website_links
import json
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI


# Load env
load_dotenv(override=True)


# Groq setup
groq_api_key = os.getenv("GROQ_API_KEY")
groq = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

link_system_prompt="""
You are provided with the lsit of links found on webpage.
You are able to decide which of the lonkswould be most relevent to include in a brochure about the company,
such as links to an about page, or a company page, or careers/jobs pages.
You should respond in json as in the example:
{
  "links":[
      {"type":"about page","url":"https://full.url/goes/here/about"},
      {"type":"career page:,"url":"https://another.full.url/goes/here/careers"}
      ]
}
"""

def get_links_user_prompt(url):
    user_prompt=f"""
Here is the list of links on the website {url}.
Please decide which of these are relevant web linksfor a brochure about the company,
respond with the full https URL in the json format.
Do not include Terms of services,Privacy,email links.

Links (some might be relative links):
"""
    links=fetch_website_links(url)
    user_prompt+="\n".join(links)
    return user_prompt


def select_relevant_links(url):
    response = groq.chat.completions.create(
       model="llama-3.1-8b-instant",  # or other Groq-supported models
       messages=[
          {"role": "system", "content": link_system_prompt},
          {"role":"user","content":get_links_user_prompt(url)}
      ],
      response_format={"type":"json_object"}
    )
    result=response.choices[0].message.content
    links=json.loads(result)
    return links

def fetch_page_all_relevent_links(url):
    contents=fetch_website_contents(url)
    relevent_links=select_relevant_links(url)
    result=f"## Landing Page\n\n{contents}\n## Relevent Links:\n"
    for link in relevent_links["links"]:
        result+=f"\n\n### Link: {link['type']}\n"
        result+=fetch_website_contents(link['url'])
    return result

brochure_system_prompt="""
You are an assistant that analyse the content of various relevant pagesfrom a company websites
and create a short brochure about the companyfor the proprective costumer,investors and recruiter.
resposnd in markdown without codeblocks.
Include the details of company culture,cutomers and career/jobs if you have the information.
"""

def get_brochure_user_prompt(company_name,url):
    user_prompt=f"""
You are looking at the company details called {company_name}
Here are the contents of the landing page and the other relevent pages;
use this content to creta a brochure of the company in markdown without code block.\n\n
"""
    user_prompt+=fetch_page_all_relevent_links(url)
    user_prompt=user_prompt[:5_000]
    return user_prompt

def create_brochure(company_name,url):
    response = groq.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": brochure_system_prompt},
        {"role":"user","content":get_brochure_user_prompt(company_name,url)}
    ],stream=True
    )

    result=""
    for chunk in response:
        result+=chunk.choices[0].delta.content or ""
        yield result

import gradio as gr
message_input_company_name=gr.Textbox(label="Enter the Company Name",info="Enter the name of company",lines=7)
message_input_url=gr.Textbox(label="Enter URL",info="Provide the url of the company",lines=7)
message_output=gr.Markdown(label="Response")

view=gr.Interface(
    fn=create_brochure,title="Brochure Generator",
    inputs=[message_input_company_name,message_input_url],
    outputs=[message_output],
    examples=[["Ed Donner","https://edwarddonner.com/"],["Tensorflow","https://tensorflow.com"]],
    flagging_mode="never"
)

view.launch()



