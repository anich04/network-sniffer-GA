"# network-sniffer-GA" 
This is the Major Project

***  Folder Structure  ***
Network-Packet-Analyzer-GA/
│
├── src/
│   ├── packet_capture.py
│   ├── extract_features.py
│   ├── analyzer.py
│   ├── utils.py
│   │
│   ├── ga_engine/
│   │   ├── ga_train.py
│   │   ├── fitness_function.py
│   │   ├── apply_ga_rules.py
│   │   └── ga_model.pkl
│   │
│   └── data/
│       ├── sample.pcap
│       ├── features.csv
│       └── ga_logs.csv
│
├── notebooks/
│   ├── GA_Training.ipynb
│   ├── Feature_Analysis.ipynb
│
├── tests/
│   ├── test_pcaps/
│   ├── evaluation_report.csv
│
├── docs/
│   ├── Project_Report.pdf
│   ├── Architecture_Diagram.png
│   ├── GA_Flowchart.png
│
├── README.md
├── requirements.txt
└── .gitignore
