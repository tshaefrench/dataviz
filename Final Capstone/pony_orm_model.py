
# coding: utf-8

# In[1]:


from pony.orm import *

db = Database()

class Job(db.Entity):
    id = PrimaryKey(int, auto=True)
    title = Required(str)
    job_description = Required(str)
    job_class = Required(str)
    
    
db.bind(provider ='sqlite', filename = '/Users/tiffanyfrench/Desktop/Final Capstone/job_postings.sqlite', create_db=True)
db.generate_mapping(create_tables=True)

