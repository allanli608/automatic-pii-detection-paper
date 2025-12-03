
LABEL_LIST = [
    "O",
    "B-EMAIL",
    "B-ID_NUM",
    "B-NAME_STUDENT",
    "B-PHONE_NUM",
    "B-STREET_ADDRESS",
    "B-URL_PERSONAL",
    "B-USERNAME",
    "I-ID_NUM",
    "I-NAME_STUDENT",
    "I-PHONE_NUM",
    "I-STREET_ADDRESS",
    "I-URL_PERSONAL",
]

label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for label, i in label2id.items()}
