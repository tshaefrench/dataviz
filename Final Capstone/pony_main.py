
# coding: utf-8

# In[1]:


from pony_orm_model import *
import csv

@db_session
def add_job(title, job_description):
    d = Job()
    d.title = title
    commit()
    d.job_description = job_description
    commit()
    d.job_class = job_class
    commit()
