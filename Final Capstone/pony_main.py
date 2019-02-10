
# coding: utf-8

# In[1]:


from pony_orm_model import *
import csv

@db_session
def add_job(title, job_description):
    d = Job()
    d.title = title
    d.job_description = job_description
    d.job_class = job_class

