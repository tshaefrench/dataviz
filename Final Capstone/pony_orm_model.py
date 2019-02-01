
# coding: utf-8

# In[ ]:


from pony.orm import *


# In[ ]:


db = Database()


# In[ ]:


class Job(db.Entity):
    id = PrimaryKey(int, auto=True)
    title = Required(str)
    job_description = Required(str)


# In[ ]:


db.bind(provider ='sqlite', filename = 'jobs.sqlite', create_db=True)
db.generate_mapping(create_tables=True)

