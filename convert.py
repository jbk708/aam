# convert_requirements.py
with open('requirements.txt') as f:
    packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]

print('dependencies = [')
for pkg in packages:
    print(f'    "{pkg}",')
print(']')