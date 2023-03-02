import sys
import yaml
import re

with open(sys.argv[1], mode='r') as file:
    environment = yaml.safe_load(file)

    pattern = '.*libgfortran.*|.*appnope.*|.*libcxx.*'

    for package in environment['dependencies']:
        if isinstance(package, str):
            if re.search(pattern, package):

                environment['dependencies'].remove(package)

                print("Match found, {} removed".format(package))

with open(sys.argv[1], mode='w') as yml:
    yaml.dump(environment, yml)