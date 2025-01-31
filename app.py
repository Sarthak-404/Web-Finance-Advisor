import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

app = Flask(__name__)
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Flask API!"


search_engine = DuckDuckGoSearchRun()
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template("""
        Summarize the content given and try to keep all the facts that are told in it.
        Try to keep it brief and knowledgeable.
        content: {result}
""")

def reply(text):
    main = prompt.invoke({'result': text})
    response = llm.invoke(main)
    return response.content


@app.route('/query', methods=['GET'])
def get_summary():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    result = search_engine.invoke(query)
    summary = reply(result)
    
    return jsonify({"query": query, "summary": summary})

if __name__ == '__main__':
    app.run(debug=True)