# Healht Care Chat Bot 


## Technique used : 
* collected the data manually in the  form of pdf's
* FAISS vector database has been used to create the embeddings for the data collected 
* Then using the langchain we are retreiving the best possible solutions for the user queries, combining it with gpt-3 providing user a proper output in a friendly manner 
* The response is then show using the Textbase chat bot UI available.

## Improvement 
* we can use opensource Large Language Model's such as llamma- 7B 13b etc.. 
* train for our specific business purpose, this would avoid the expense of using open-ai api for the each user, and reduce overall cost and be helpful in long term. 
