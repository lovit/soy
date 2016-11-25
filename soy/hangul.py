kor_begin     = 44032
kor_end       = 55199
chosung_base  = 588
jungsung_base = 28

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def normalize(doc, english=False, number=False, punctuation=False, remains={}):
    
    f = ''
    
    for c in doc:

        i = ord(doc)
        
        if c == ' ':
            f += ' '
        
        elif (i >= kor_begin) and (i <= kor_end):
            f += c
            
        elif (english) and ( (i >= 97 and i <= 122) or (i >= 65 and i <= 90) ):
            f += c
            
        elif (number) and (i >= 48 and i <= 57):
            f += c
        
        elif (punctuation) and (i == 33 or i == 34 or i == 39 or i == 44 or i == 46 or i == 63 or i == 96):
            f += c
            
        elif c in remains:
            f += c
            
    return f


def split_jamo(c):
    
    base = ord(c)
    
    if base < kor_begin or base > kor_end:
        return None
    
    base -= kor_begin
    
    cho  = base // chosung_base
    jung = ( base - cho * chosung_base ) // jungsung_base 
    jong = ( base - cho * chosung_base - jung * jungsung_base )
    
    return [chosung_list[cho], jungsung_list[jung], jongsung_list[jong]]


def combine_jamo(chosung, jungsung, jongsung):
    return chr(kor_begin + chosung_base * chosung_list.index(chosung) + jungsung_base * jungsung_list.index(jungsung) + jongsung_list.index(jongsung))
