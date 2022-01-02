#두개의 for문을 중첩해서 1,2,3,4 나는 숫자가 5줄 나오게 출력해라

for i in range(5):
    for j in range(1,5):
        print(j,end="")
    print("")

#구구단 2단
for i in range(1,10):
    print('{}*{}={}'.format(2,i,2*i))

#구구단 전체
for j in range(2,10):
    for i in range(1,10):
        print('{}*{}={}'.format(j,i,j*i))
    print()    

#a~b 구구단
a,b = map(int,input().split()) #map함수는 여러번 적용하고 싶을떄, 앞쪽에 적용할 함수를 넣어준다

for i in range(2,10):
    for j in range(1,10):
        print('{}*{}={}'.format(j,i,j*i))
    print()   

#주사위 두개 던졌을떄 모든 경우의 수
for i in range(1,7):
    for j in range(1,7):
        print(i,j)


