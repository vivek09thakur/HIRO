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


def display_list(list):
    for index, item in enumerate(list):
        print(index, ")", item)


def type_input(message, datatype):
    while True:
        try:
            user_input = datatype(input(message))
            return user_input
        except ValueError:
            print("Invalid input")
            continue


def menu_selector(option_arr):
    while True:
        print("Select an option:")
        display_list(option_arr)
        option = type_input("Enter option number: ", int)
        if option < 0 or option >= len(option_arr):
            print("Invalid option")
            continue
        return option_arr[option]


def ask_filter_arr(arr, message_function):
    filter_arr = []
    for elem in arr:
        message = message_function(elem)
        print(message)
        if ask_yes_or_no():
            filter_arr.append(elem)
    return filter_arr


def ask_yes_or_no():
    while True:
        inp = input("[yes or no] : ")

        if inp == "yes":
            return True
        if inp == "no":
            return False

        print("provide proper answers i.e. (yes/no) : ", end="")
