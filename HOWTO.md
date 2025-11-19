# 1. Generate the dataset with flow rates
python generate_kenya_dataset.py

# Expected output: 17,000 records with realistic flow rates
# Drip: ~30% of irrigation events
# Sprinkler: ~40% of irrigation events  
# Manual: ~30% of irrigation events
# None: ~25% no irrigation needed

# 2. Train model (flow rate not used in detection, just for monitoring)
python train_model.py

# 3. Start backend - now tracks valve flow rates
python app.py

# 4. Check dashboard

## 📊 What You'll See

**Console output** from generator:
```
Irrigation Flow Rates (based on industry standards):
  Drip         -   2.5 L/min (±0.8) - Drip irrigation (efficient, low flow)
  Sprinkler    -  15.0 L/min (±3.5) - Sprinkler irrigation (moderate flow)
  Manual       -  35.0 L/min (±8.0) - Manual/hose irrigation (high flow)
```

**Dashboard valve card** when active:
```
🚰 Irrigation Valve
   OPEN
   Flow: 15.23 L/min
   Status: Active