from beta.HIRO import HIRO

hiro = HIRO()
hiro.prepare()

if __name__=='__main__':
    patient_name = input('[patient name] :: ')
    hiro.introduce(patient_name)
    # get user's problem
    user_problem = input('\n[can you explain your problem] :: ')
    result = hiro.get_user_problem(user_problem)
    print(f'\n[results] :: {result}')