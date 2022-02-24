#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
import joblib
import asyncio


# In[2]:


app = Flask(__name__) #to make sure it doesn't run library but your own file


# In[3]:


from flask import request, render_template 

@app.route("/", methods = ["GET", "POST"]) #deocrator aka make sure run this before run the codes below. A function before a function. 
def index(): #can name it anything, doesn't need to be index
    asyncio.set_event_loop(asyncio.new_event_loop())
    if request.method == "POST": 
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        print(income, age, loan)
        model = joblib.load("creditcarddefault2")
        pred = model.predict([[float(income), float(age), float(loan)]])
        print(pred)
        pred= pred[0]
        s= "The predicted default is " + str(pred)
        
        
        return(render_template("index.html", result =s))
    else: 
        return(render_template("index.html", result = "Predict 1"))


# In[ ]:


#only run if its your programme
if __name__ == "__main__":
    app.run()


# In[ ]:





# In[ ]:





# In[ ]:




