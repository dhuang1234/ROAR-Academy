import os

source_filename = 'motto.txt'

try:
    path = os.path.dirname(os.path.abspath(__file__))
    s = open(path+'/'+source_filename,'w')

    s.write("Fiat Lux!")
    s.close()

    s1 = open(path+'/'+source_filename,'a+')
    s1.seek(0)
    print("before adding 'let there be light'")
    print(s1.read())

    s1.write("\nLet there be light!")
    s1.seek(0)
    print("\nafter adding 'let there be light'")
    print(s1.read())

except IOError:
    print('IO Error! Please check valid file names and paths')
    exit
finally:
    s1.close()

