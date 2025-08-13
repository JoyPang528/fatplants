1. Open a DEDICATED Terminal Window for the Flask Server:
This terminal will only run your Flask app. Do not close it.
Activate your Conda environment: conda activate citation_env
Navigate to the directory where your citation_api.py and citation_retrieval_module_python.py files are located.
Run the Flask app: python citation_api.py
CRITICAL VERIFICATION: Look at the output in this terminal. You must see lines similar to these, indicating it's running:
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5001
* Running on http://[YOUR_LOCAL_IP]:5001
Press CTRL+C to quit
If you see any error messages here (e.g., "Address already in use," "Permission denied," or if the prompt returns immediately), the Flask server is NOT running. You must resolve that error first.

2. Open a SEPARATE Terminal Window for the Chatbot:
This terminal will run your chatbot.
Activate your Conda environment: conda activate citation_env
Navigate to the directory where your chatbot_test python script is located.
Set the necessary environment variables for the chatbot:

Run the chatbot script: python chatbot_test.py
