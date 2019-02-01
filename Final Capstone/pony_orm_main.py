
# coding: utf-8

# In[ ]:


from pony_orm_model import *
import csv

@db_session
def add_job(title, job_description):
    d = Job()
    d.title = title
    d.job_description = job_description

with open('jobs.csv', 'r') as fd:
    reader = csv.DictReader(fd)
    for line in reader:
        add_job(line['title'], line['job_description'])
        

