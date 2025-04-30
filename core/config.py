# Default filter parameters
DEFAULT_FILTER_PARAMS = {
    "fc_l": 150.0,
    "fc_h": 5,
    "orde_filter": 2,
}

# Default thresholds for peak detection with sample 1min
# DEFAULT_THRESHOLDS = {
#     "P": "0.11 0.2",
#     "Q": "-0.28 -0.22",
#     "R": "1 1.5",
#     "S": "-0.9 -0.50",
#     "T": "0.23 0.38",
# }
# # Default thresholds for peak detection with sample 10sec 150
DEFAULT_THRESHOLDS = {
    "P": "15 21",
    "Q": "-35 -25",
    "R": "115 140",
    "S": "-85 -60",
    "T": "32 43",
}
# Default thresholds for peak detection with sample 10sec 100hz
# DEFAULT_THRESHOLDS = {
#     "P": "5 9",
#     "Q": "-35 -25",
#     "R": "38 45",
#     "S": "-85 -60",
#     "T": "15 18",
# }

DEFAULT_SAMPLE_INTERVAL = 0.01 #seconds