import sys
print(f"Python: {sys.executable}")
if "envs/building_energy" in sys.executable:
    print("✅ IN BUILDING_ENERGY ENVIRONMENT")
else:
    print("❌ NOT IN ENVIRONMENT")
