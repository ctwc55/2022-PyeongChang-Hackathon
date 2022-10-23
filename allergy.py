def detect(food_list):
    print(f'모듈을 불러왔다!!! {food_list}')


    if 'almondbreeze' in food_list:
        return '우유 알레르기와 견과류 알레르기에 주의하십시오.'
    elif 'cheesebagel' in food_list:
        return '우유 알레르기와 밀 알레르기에 주의하십시오.'
    elif ('ggobukchip' in food_list or 'goraebob' in food_list or 
            'ohpotato' in food_list or  'pepero' in food_list):
        return '계란 알레르기와 밀 알레르기에 주의하십시오.'
    elif 'plum' in food_list:
        return '과일 알레르기에 주의하십시오.'
    elif 'minicrap' in food_list:
        return '갑각류 알레르기에 주의하십시오.'
    elif 'chocoleta' in food_list:
        return '우유와 커피와는 같이 드시지 마시오.'
    elif 'jinro' in food_list:
        return '술은 몸에 좋지 않습니다.'
    elif 'chocoleta' in food_list and 'almondbreeze' in food_list:
        return '초콜릿과 우유는 상극인 음식입니다.'
    elif "pepero" in food_list and "almondbreeze" in food_list:
        return '빼빼로와 아몬드는 상극인 음식입니다.'
