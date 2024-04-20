import os
import json
json_file = 'sample.json'
with open(json_file, 'r') as f:
    content =json.load(f)
# str to dict
object_info = content['annotation']['object']
for class_info in object_info :
    print(class_info)
    print(f' -------------------------------------------------------------- ')


#semantic_info = content['object']
#print(semantic_info)