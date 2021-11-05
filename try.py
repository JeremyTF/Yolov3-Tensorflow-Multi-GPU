type_dict = {}

print(type_dict)

i = 0


# print(type_dict)
from core.config import cfg

with open(cfg.YOLO.CLASSES, 'r') as data:
    for ID, name in enumerate(data):
        name = name.strip('\n')
        type_dict[ID] = 0

type_dict[0] +=100
print(type_dict)