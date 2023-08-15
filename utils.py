import re


def check_list_match(list, value):
    pred_list = []  # prediction list
    value = value.replace(" ", "_")
    patt = f"{value}"
    regexp = re.compile(patt)
    pred_list = [item for item in list if regexp.search(item)]
    if len(pred_list) > 0:
        return pred_list
    else:
        return []
