__all__ = ['out', 'CSV','CSVW','summx','submx','multchmx','multmx','trnsmx','dettmx','summ','subb','multch','mult','trns','mtx1','mtx_gnrt','Matrixx','inverse','SLAU_mtx','Coef','SLAU_gnrt','Coef_gnrt','yacoby','GousJ','GousJFR','proverka','SLAU','int_lagr','int_stnd','aprx_lin','aprx_quad','aprx_nrmDist','Int_Aprx','velvet','ODU','mtx_Way','mtxcreat','rndWay','summ_Way','Optimize']

"""
def out(a): 
[Красивый вывод матрицы]

Args:
    a ([list]): [Матрица]

def CSV(): 
[Считываем Матрицу из CSV файла]
-Вводится название файла

Returns:
    [list]: [Матрица]

def CSVW(a): 
[Запись в CSV файл]
-Вводится название файла
Args:
    a ([list]): [Матрица]

def summx(a,b): 
[Сложение матриц с использованием numpy]

Args:
    a ([list]): [Матрица]
    b ([list]): [Матрица]

Returns:
    [list]: [Результат]

def submx(a,b): 
[Вычитание матриц с использованием numpy]

Args:
    a ([list]): [Матрица]
    b ([list]): [Матрица]

Returns:
    [list]: [Результат]

def multchmx(a,c): 
[Умножение матрицs на число с использованием numpy]

Args:
    a ([list]): [Матрица]
    c ([float]): [Число]

Returns:
    [type]: [description]

def multmx(a,b): 
[Умножение матриц с использованием numpy]

Args:
    a ([list]): [Матрица]
    b ([list]): [Матрица]

Returns:
    [list]: [Результат]

def trnsmx(a): 
[Транспонирование Матрицы с использованием numpy]

Args:
    a ([list]): [Матрица]

Returns:
    [list]: [Матрица]

def dettmx(a): 
[Поиск определителя матрицы с использованием numpy]

Args:
    a ([list]): [Матрица]

Returns:
    [float]: [Определитель]

def summ(a,b): 
[Сложение матриц]

Args:
    a ([list]): [Матрица]
    b ([list]): [Матрица]

Returns:
    [list]: [Результат]

def subb(a,b): 
[Вычитание матриц]

Args:
    a ([list]): [Матрица]
    b ([list]): [Матрица]

Returns:
    [list]: [Результат]

def multch(a,c): 
[Умножение матрицы на число]

Args:
    a ([list]): [Матрица]
    c ([float]): [Число]

Returns:
    [type]: [description]

def mult(a,b):
[Умножение матриц]

Args:
    a ([list]): [Матрица]
    b ([list]): [Матрица]

Returns:
    [list]: [Результат]

def trns(a):  
[Транспонирование Матрицы]

Args:
    a ([list]): [Матрица]

Returns:
    [list]: [Матрица]

def mtx1():
[Создание матрицы]
-Пользователь вводит кол-во строк/столбцов/Элементы матрицы
Returns:
    [list]: [Матрица]

def mtx_gnrt(): 
[Ручная генерация матрицы]
- Пользователь вводит кол-во строк/столбцов
- Пользователь вводит диапазон генерации
Returns:
    [list]: [Сгенерированная матрица]

def Matrixx():
[Запуск программы по работе с Матрицами]

def inverse(num): 
[Вычисление обратной матрицы]

Args:
    num ([list]): [Матрица]

Returns:
    [list]: [Обратная матрица]

def SLAU_mtx():
[Ввод матрицы СЛАУ]
- Пользователь вводит ранг матрицы
- Пользователь вводит значения матрицы поэлементно
    Returns:
        [list]: [Матрица СЛАУ]

def Coef():  
[Ввод столбца свободных членов]
-Пользователь вводит значения поэлементно
Returns:
    [list]: [Массив свободных членов]

def SLAU_gnrt(): 
[Ручная генерация матрицы СЛАУ]
- Пользователь вводит ранг матрицы
- Пользователь вводит диапазон значений
Returns:
    [list]: [Матрица СЛАУ]

def Coef_gnrt():  
[Генерация массива свободных членов]
- Пользователь вводит диапазон значений
Returns:
    [list]: [Массив свободных членов]

def yacoby(a1):
[Метод Якоби для решения СЛАУ]

Args:
    a1 ([list]): [Сшитая матрица(Матрица СЛАУ + столбец коэф.)]

Returns:
    [list]: [Решение]

def GousJ(num): 
[Метод Жордана-Гауса для решения СЛАУ]

Args:
    num ([list]): [Сшитая матрица(Матрица СЛАУ + столбец коэф.)]

Returns:
    [list]: [Решение]

def GousJFR(num): 
[Метод Жордана-Гауса для решения СЛАУ с типом Fraction]

Args:
    num ([list]): [Сшитая матрица(Матрица СЛАУ + столбец коэф.)]

Returns:
    [list]: [Решение в типе Fraction]

def proverka(num,num1):
[Проверка и сшивка СЛАУ]
-Проверяет возможно ли решить СЛАУ
-Сшивает матрицу(добавляет столбец коэф-в с матрицу СЛАУ)

Args:
    num ([list]): [Матрица СЛАУ]
    num1 ([list]): [Массив свободных Коэф-в]

Returns:
    [list]: [Сшитая СЛАУ]

def SLAU():
[Запуск программы по решению СЛАУ]

def int_lagr(a,b): 
[Интерполяция Лагранжем]

Args:
    a ([list]): [Массив значений по X]
    b ([list]): [Массив значений по Y]

Returns:
    [list]: [Решение]

def int_stnd(a,b):
[Интерполяция Ньютоном]

Args:
    a ([list]): [Массив значений по X]
    b ([list]): [Массив значений по Y]

Returns:
    [list]: [Решение]

def aprx_lin(a,b): 
[Аппроксимация линейной функцией]

Args:
    a ([list]): [Массив значений по X]
    b ([list]): [Массив значений по Y]

Returns:
    [list]: [Решение]

def aprx_quad(a,b):  
[Аппроксимация квадратичной функцией]

Args:
    a ([list]): [Массив значений по X]
    b ([list]): [Массив значений по Y]

Returns:
    [list]: [Решение]

def aprx_nrmDist(a,b): 
[Аппроксимация норм. Распределения]

Args:
    a ([list]): [Массив значений по X]
    b ([list]): [Массив значений по Y]

Returns:
    [list]: [Решение]

def Int_Aprx():
[Запуск программы по работе с Интерполяцией/Апроксимацией]

def velvet():
[Запуск программы по работе с Вейвлет/Фурье Анализом]
- Запускает интерфейс программы
- все функции внутри программы работают через функции библиотек python

def ODU():
[Запуск программы по работе с ОДУ]

def mtx_Way(a):
[Генерация матрицы весовых коэф-в]
-Используется для решения задач на оптимизацию
-Диапазон генерации 1-50
Args:
    a ([int]): [Ранг матрицы]

Returns:
    [list]: [Матрица весовых коэф-в]

def mtxcreat():
[Создание матрицы весовых коэф-в]
-Пользователь вводит количество городов(ранг матрицы)
-Пользователь вводит значение поэлементно
Returns:
    [list]: [Матрицы весовых коэф-в]

def rndWay(a): 
[Случайный путь]

Args:
    a ([list]): [Матрица весовых коэф-в]

Returns:
    [list]: [Случайный путь]

def summ_Way(m1,way): # Длина пути
[Длина пути]

Args:
    m1 ([list]): [Матрица весовых коэф-в]
    way ([list]): [Путь]

Returns:
    [float]: [Длина пути]

def Optimize():
[Запуск программы по Оптимизации]
"""
import numpy as np
from random import randint, shuffle, sample, choices, uniform
import csv
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
#
import math
import copy
from numpy import linalg as LA
from fractions import Fraction
#
from math import *
from scipy.interpolate import make_interp_spline, BSpline
#
from sympy import *
from scipy.fft import fft, ifft
import pywt
#
import scipy
#
from math import exp
#
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

def out(a): # Красивый вывод матрицы
    """[Красивый вывод матрицы]

    Args:
        a ([list]): [Матрица]
    """
    try:
        for i in a:
            print(i)
        print('\n')
    except TypeError:
        return 0
#------------------ 2 --------------------#
def csv_trns():
    try:
        with open(input('Введите в формате: "Название"."csv"\n'), encoding='utf-8-sig') as data_file:
            csv = []
            A = []
            for line in data_file:
                csv=line.strip().split(';')
                A.append(csv)
            out(A)
        return A
    except FileNotFoundError:
        print('Искомый файл не найден!!!\n')
        return csv_trns()
def CSV(): # Считываем Матрицу из CSV файла
    """[Считываем Матрицу из CSV файла]
    -Вводится название файла
    
    Returns:
        [list]: [Матрица]
    """
    try:
        with open(input('Введите в формате: "Название"."csv"\n'), encoding='utf-8-sig') as data_file:
            csv = []
            A = []
            for line in data_file:
                csv=line.strip().split(';')
                csv = [(x) for x in csv]
                A.append(csv)
            print('Вы импортировали:')
            out(A)
        return A
    except FileNotFoundError:
        print('Искомый файл не найден!!!\n')
        return CSV()
def CSVW(a): #Запись в CSV файл
    """[Запись в CSV файл]
    -Вводится название файла
    Args:
        a ([list]): [Матрица]
    """
    with open(input('Введите в формате: "Название"."csv"\n'), mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter = ";", lineterminator="\r")
        for i in a:
            file_writer.writerow(i)
def zapis(k):
    a = input('Записать матрицу в CSV файл?\n1)ДА\n2)НЕТ\n')
    if a == '1':
        CSVW(k)

####### ФУНКЦИИ NUMPY
def summx(a,b): #Сложение
    """[Сложение матриц с использованием numpy]

    Args:
        a ([list]): [Матрица]
        b ([list]): [Матрица]

    Returns:
        [list]: [Результат]
    """
    v = time.time()
    a = np.array(a)
    b = np.array(b)
    try:
        smx = a + b
        if (time.time()-v)>0.0000001:
            print(f"Время работы программы: {time.time()-v} секунд")
        else:
            print("Слишком быстро, начальника, мы не успеваем считать!")
        return smx
    except ValueError:
        print('\n>>> Матрицы разного ранга!!!')    
def submx(a,b): #Вычитание
    """[Вычитание матриц с использованием numpy]

    Args:
        a ([list]): [Матрица]
        b ([list]): [Матрица]

    Returns:
        [list]: [Результат]
    """
    v = time.time()
    a = np.array(a)
    b = np.array(b)
    try:
        smx = a + b
        if (time.time()-v)>0.0000001:
            print(f"Время работы программы: {time.time()-v} секунд")
        else:
            print("Слишком быстро, начальника, мы не успеваем считать!")
        return smx
    except ValueError:
        print('\n>>> Матрицы разного ранга!!!')
def multchmx(a,c): #Умножение на число
    """[Умножение матрицs на число с использованием numpy]

    Args:
        a ([list]): [Матрица]
        c ([float]): [Число]

    Returns:
        [type]: [description]
    """
    v = time.time()
    a = np.array(a)
    smx = a * c
    if (time.time()-v)>0.0000001:
        print(f"Время работы программы: {time.time()-v} секунд")
    else:
        print("Слишком быстро, начальника, мы не успеваем считать!")
    return smx
def multmx(a,b): #Перемножение двух матриц
    """[Умножение матриц с использованием numpy]

    Args:
        a ([list]): [Матрица]
        b ([list]): [Матрица]

    Returns:
        [list]: [Результат]
    """
    v = time.time()
    a = np.array(a)
    b = np.array(b)
    if len(a)!=len(b[0]):
        print("\n>>> Матрицы не могут быть перемножены!!!")
    else:
        mulmx = a.dot(b)
        if (time.time()-v)>0.0000001:
            print(f"Время работы программы: {time.time()-v} секунд")
        else:
            print("Слишком быстро, начальника, мы не успеваем считать!")
        return mulmx
def trnsmx(a): #Транспонирование 
    """[Транспонирование Матрицы с использованием numpy]

    Args:
        a ([list]): [Матрица]

    Returns:
        [list]: [Матрица]
    """
    v = time.time()
    a = np.array(a)
    trns = a.transpose()
    if (time.time()-v)>0.0000001:
        print(f"Время работы программы: {time.time()-v} секунд")
    else:
        print("Слишком быстро, начальника, мы не успеваем считать!")
    return trns
def dettmx(a): #Поиск определителя
    """[Поиск определителя матрицы с использованием numpy]

    Args:
        a ([list]): [Матрица]

    Returns:
        [float]: [Определитель]
    """
    v = time.time()
    a = np.array(a)
    d = np.linalg.det(a)
    if (time.time()-v)>0.0000001:
        print(f"Время работы программы: {time.time()-v} секунд")
    else:
        print("Слишком быстро, начальника, мы не успеваем считать!")
    return round(d)

####### СВОИ ФУНКЦИИ
def summ(a,b): #Сложение
    """[Сложение матриц]

    Args:
        a ([list]): [Матрица]
        b ([list]): [Матрица]

    Returns:
        [list]: [Результат]
    """
    v = time.time()
    try:
        m3 = []
        for i in range(len(a)):
            t = [] #Стока матрицы(буфер)
            for j in range(len(b[i])):
                s = a[i][j]+b[i][j]
                t.append(s)
            m3.append(t)
        if (time.time()-v)>0.0000001:
            print(f"Время работы программы: {time.time()-v} секунд")
        else:
            print("Слишком быстро, начальника, мы не успеваем считать!")
        return m3
    except IndexError:
        print('\n>>> Матрицы разного ранга')  
def subb(a,b): #Вычитание
    """[Вычитание матриц]

    Args:
        a ([list]): [Матрица]
        b ([list]): [Матрица]

    Returns:
        [list]: [Результат]
    """
    v = time.time()
    try:
        m3 = []
        for i in range(len(a)):
            t = [] #Стока матрицы(буфер)
            for j in range(len(b[i])):
                s = a[i][j]-b[i][j]
                t.append(s)
            m3.append(t)
        if (time.time()-v)>0.0000001:
            print(f"Время работы программы: {time.time()-v} секунд")
        else:
            print("Слишком быстро, начальника, мы не успеваем считать!")
        return m3
    except IndexError:
        print('\n>>> Матрицы разного ранга!!!')
def multch(a,c): #Умножение на число
    """[Умножение матрицы на число]

    Args:
        a ([list]): [Матрица]
        c ([float]): [Число]

    Returns:
        [type]: [description]
    """
    v = time.time()
    m3 = []
    for i in range(len(a)):
        t = [] #Стока матрицы(буфер)
        for j in range(len(a[i])):
            s = a[i][j]*c
            t.append(s)
        m3.append(t)
    if (time.time()-v)>0.0000001:
        print(f"Время работы программы: {time.time()-v} секунд")
    else:
        print("Слишком быстро, начальника, мы не успеваем считать!")
    return m3
def mult(a,b):
    """[Умножение матриц]

    Args:
        a ([list]): [Матрица]
        b ([list]): [Матрица]

    Returns:
        [list]: [Результат]
    """
    v = time.time()
    s=0     #сумма
    t=[]    #буфер
    m3=[] # конечная матрица
    if len(b)!=len(a[0]):
        print("\n>>> Матрицы не могут быть перемножены!!!")
    else:
        for z in range(0,len(a)):
            for j in range(0,len(b[0])):
                for i in range(0,len(a[0])):
                    s=s+a[z][i]*b[i][j] #Считаем сумму нового значения
                t.append(s) # Записываем сумму в буфер 
                s=0 #Обнуляем сумму для расчета новой строки матрицы
            m3.append(t) #Переносим значение из буфера в новую матрицу
            t=[] #Обнуляем буфер
        if (time.time()-v)>0.0000001:
            print(f"Время работы программы: {time.time()-v} секунд")
        else:
            print("Слишком быстро, начальника, мы не успеваем считать!")
        return m3
def trns(a):  #Транспонирование
    """[Транспонирование Матрицы]

    Args:
        a ([list]): [Матрица]

    Returns:
        [list]: [Матрица]
    """
    trns = [[0 for j in range(len(a))] for i in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            trns[j][i] = a[i][j]
    return trns
def determinant(matrix, mul): #Поиск определителя
    mat2 = matrix
    width = len(mat2)
    if width == 1:
        return mul * mat2[0][0]
    else:
        sign = -1
        answer = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(mat2[j][k])
                m.append(buff)
            sign *= -1
            answer = answer + mul * determinant(m, sign * mat2[0][i])
    return answer

##### ФУНКЦИИ СОЗДАНИЯ МАТРИЦ
def mtx1():
    """[Создание матрицы]
    -Пользователь вводит кол-во строк/столбцов/Элементы матрицы
    Returns:
        [list]: [Матрица]
    """
    def cock():  #Создание матрицы 1
        try:
            num[i][j] = int(input())
            return num[i][j]
        except ValueError:
            print('Ошибка ввода!!!\nВведите элемент заново\n')
            return cock()
        
    try:
        x = int(input('Матрица A:\nВведите кол-во строк: '))
        y = int(input('Введите кол-во столбцов: '))
        num = [[0 for n in range(y)] for nn in range(x)]
        print('Введите значения матрицы поэлементно')
        for i in range(x):
            for j in range(y):
                cock()
        print('Вы ввели матрицу:')
        out(num) 
        return num 
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново кол-во сток/столбцов\n')
        return(mtx1())
def mtx2():  #Создание матрицы 2
    def cock():
        try:
            num1[i][j] = int(input())
            return num1[i][j]
        except ValueError:
            print('Ошибка ввода!!!\nВведите элемент заново\n')
            return cock()
    try:
        x = int(input('Матрица B:\nВведите кол-во строк: '))
        y = int(input('Введите кол-во столбцов: '))
        num1 = [[0 for n in range(y)] for nn in range(x)]
        print('Введите значения матрицы поэлементно')
        for i in range(x):
            for j in range(y):
                cock()
        print('Вы ввели матрицу:')
        out(num1) 
        return num1 
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново кол-во сток/столбцов\n')
        return(mtx2())
def mtx_gnrt(): #Ручная генерация матрицы
    """[Ручная генерация матрицы]
    - Пользователь вводит кол-во строк/столбцов
    - Пользователь вводит диапазон генерации
    Returns:
        [list]: [Сгенерированная матрица]
    """
    mtx = []
    try:
        strr = int(input('Введите кол-во строк>>> '))
        coll = int(input('Введите кол-во столбцов>>> '))
        minn = int(input('Введите диапазон значений:\nВведите минимальное значение>>> ')) 
        maxx = int(input('Введите максимальное значение>>> '))
        for i in range(strr):
            row = []
            for j in range(coll):
                gen_num = randint(minn,maxx)
                row.append(gen_num)
            mtx.append(row)
        print('Сгенирированная матрица = ')
        out(mtx)
        return mtx
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново\n')
        return mtx_gnrt()

##### ФУНКЦИИ ПОДСЧЕТА ВРЕМЕНИ
def dettmxTIME(a): #Поиск определителя
    v = time.perf_counter()
    a = np.array(a)
    d = np.linalg.det(a)
    vv = time.perf_counter()-v
    print('Определитель через numpy = ',round(d))
    return vv
def determinantTIME(matrix, mul): #Поиск определителя
    v = time.perf_counter()
    mat2 = matrix
    width = len(mat2)
    if width == 1:
        return mul * mat2[0][0]
    else:
        sign = -1
        answer = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(mat2[j][k])
                m.append(buff)
            sign *= -1
            answer = answer + mul * determinantTIME(m, sign * mat2[0][i])
    return answer
def trnsmxTIME(a): # время Транспонирования numpy
        v = time.perf_counter()
        trns = a.transpose()
        vv = time.perf_counter()-v
        print('Транспонированная матрица:\n', trns)
        return vv
def trnsTIME(a):  # время Транспонирования своей ф-цией
        v = time.perf_counter()
        trns = [[0 for j in range(len(a))] for i in range(len(a[0]))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                trns[j][i] = a[i][j]
        vv = time.perf_counter()-v
        print('Транспонированная матрица:')
        out(trns)
        return vv

def dop_zadanie(): # Функция доп задания
    try:
        global detnptime
        global o1
        global transnptime
        global transtime
        global o2
        global dettime
        transtime = []
        transnptime =[]
        dettime =[]
        detnptime = []
        o1=[]
        o2=[]
        minn = int(input('Введите диапазон значений:\nВведите минимальное значение>>> ')) 
        maxx = int(input('Введите максимальное значение>>> '))+1
        x = 1
        for i in range(x):
            while x  <= 10:
                y = 1
                for j in range(y):
                    while y <=10:
                        rnd_mtx = np.array(np.random.randint(minn,maxx, (x,y)))
                        print('Сгенирированная матрица',x,' на ',y,':')
                        print(rnd_mtx,'\n')
                        o1.append(x*y)
                        if x == y:
                            o2.append(x*y)
                            t = dettmxTIME(rnd_mtx)
                            print(f"Время поиска определителя: {t} секунд\n")
                            detnptime.append(t)
                            v = time.perf_counter()
                            print('Определитель через через свою ф-цию = ',determinantTIME(rnd_mtx,1),)
                            vv = time.perf_counter()-v
                            print(f"Время поиска определителя: {vv} секунд\n")
                            dettime.append(vv)
                            t = trnsmxTIME(rnd_mtx)
                            print(f"Время транспонирования через numpy: {t} секунд\n")
                            transnptime.append(t)
                            t = trnsTIME(rnd_mtx)
                            print(f"Время транспонирования через свою ф-цию: {t} секунд\n")
                            transtime.append(t)
                            print(''.center(36,'_'))
                        else:
                            t = trnsmxTIME(rnd_mtx)
                            print(f"Время транспонирования через numpy: {t} секунд\n")
                            transnptime.append(t)
                            t = trnsTIME(rnd_mtx)
                            print(f"Время транспонирования через свою ф-цию: {t} секунд\n")
                            transtime.append(t)
                            print(''.center(36,'_'))
                        y = y + 1
                x = x + 1    
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново\n')
        return dop_zadanie()

#--- ОСНОВНОЙ КОД РЕШЕНИЯ МАТРИЦ ---#
def Matrixx():
    """[Запуск программы по работе с Матрицами]
    """
    while (True):
        inpt = input('\nВыберите метод ввода матриц:\n1)Ввод вручную\n2)Импорт CSV файла\n3)Случайная генерация\n4)Использовать шаблоны\n5)Сравнение скорости работы ф-ций(доп задание)\nВведите любой другой символ для выхода из программы\n>>>  ')
        if inpt == '1':
            while(True):
                cmd = input('\nВыберите операцию:\n1)Сложение\n2)Вычитание\n3)Умножение на число\n4)Умножение\n5)Траспонирование\n6)Вычисление определителя\n0)Выбрать другой способ ввода\n\nВведите любой другой символ для выхода из программы\n>>>  ')
                if cmd == '1':
                    num = mtx1()
                    num1 = mtx2()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = summ(num, num1)
                    elif sposob == '2':
                        res = summx(num,num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '2':
                    num = mtx1() 
                    num1 = mtx2()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = subb(num, num1)
                    elif sposob == '2':
                        res = submx(num,num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '3':
                    num = mtx1()
                    kk = int(input('Введите число: '))
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = multch(num, kk)
                    elif sposob == '2':
                        res = multchmx(num,kk)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '4':
                    num = mtx1() 
                    num1 = mtx2()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = mult(num, num1)
                    elif sposob == '2':
                        res = multmx(num, num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '5':
                    num = mtx1()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = trns(num)
                    elif sposob == '2':
                        res = trnsmx(num)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '6':
                    num = mtx1()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        v = time.time()
                        res = determinant(num,1)
                        if (time.time()-v)>0.0000001:
                            print(f"Время работы программы: {time.time()-v} секунд")
                        else:
                            print("Слишком быстро, начальника, мы не успеваем считать!")
                    elif sposob == '2':
                        print('Определитель = ',dettmx(num))
                    else:
                        print('Неправильная команда!!!')
                        break  
                elif cmd == '0':
                    break
                if cmd == '1' or cmd == '2' or cmd == '3' or cmd == '4' or cmd == '5':
                    #Проверка на ошибки
                    try:
                        if res != None:
                            print('\nПолученная матрица: ')
                            out(res)
                            zapis(res)
                        else:
                            print('Введите заново\n')
                    except ValueError:
                        print('\nПолученная матрица: ')
                        out(res)
                        zapis(res)
                elif cmd == '6':
                    print('Определитель = ',res,'\n')
                else:
                    print('Программа остановлена')
                    exit()
        elif inpt == '2':
            while(True):
                cmd = input('\nВыберите операцию:\n1)Сложение\n2)Вычитание\n3)Умножение на число\n4)Умножение\n5)Траспонирование\n6)Вычисление определителя\n0)Выбрать другой способ ввода\n\nВведите любой другой символ для выхода из программы\n>>> ')
                if cmd == '1':
                    print('Введите название первого файла: ')
                    num = CSV()
                    print('Введите название второго файла: ')
                    num1 = CSV()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = summ(num, num1)
                    elif sposob == '2':
                        res = summx(num,num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '2':
                    print('Введите название первого файла: ')
                    num = CSV()
                    print('Введите название второго файла: ')
                    num1 = CSV()
                    if sposob == '1':
                        res = subb(num, num1)
                    elif sposob == '2':
                        res = submx(num,num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                    res = subb(num, num1)
                elif cmd == '3':
                    print('Введите название файла: ')
                    num = CSV()
                    kk = int(input('Введите число: '))
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = multch(num, kk)
                    elif sposob == '2':
                        res = multchmx(num,kk)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '4':
                    print('Введите название первого файла: ')
                    num = CSV()
                    print('Введите название второго файла: ')
                    num1 = CSV()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = mult(num, num1)
                    elif sposob == '2':
                        res = multmx(num, num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '5':
                    print('Введите название файла: ')
                    num = csv_trns()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = trns(num)
                    elif sposob == '2':
                        res = trnsmx(num)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '6':  
                    print('Введите название файла: ')
                    num = CSV()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        v = time.time()
                        res = determinant(num,1)
                        if (time.time()-v)>0.0000001:
                            print(f"Время работы программы: {time.time()-v} секунд")
                        else:
                            print("Слишком быстро, начальника, мы не успеваем считать!")
                    elif sposob == '2':
                        print('Определитель = ',dettmx(num))
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '0':
                    break
                if cmd == '1' or cmd == '2' or cmd == '3' or cmd == '4' or cmd == '5':
                    #Проверка на ошибки
                    try:
                        if res != None:
                            print('\nПолученная матрица: ')
                            out(res)
                            zapis(res)
                        else:
                            print('Введите заново\n')
                    except ValueError:
                        print('\nПолученная матрица: ')
                        out(res)
                        zapis(res)
                elif cmd == '6':
                    print('Определитель = ',res,'\n')
                else:
                    print('Программа остановлена')
                    exit()
        elif inpt == '3':
            while(True):
                cmd = input('\nВыберите операцию:\n1)Сложение\n2)Вычитание\n3)Умножение на число\n4)Умножение\n5)Траспонирование\n6)Вычисление определителя\n0)Выбрать другой способ ввода\n\nВведите любой другой символ для выхода из программы\n>>> ')
                if cmd == '1':
                    print('\nГенерация первой матрицы')
                    num = mtx_gnrt()
                    print('\nГенерация второй матрицы')
                    num1 = mtx_gnrt()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = summ(num, num1)
                    elif sposob == '2':
                        res = summx(num,num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '2':
                    print('\nГенерация первой матрицы')
                    num = mtx_gnrt()
                    print('\nГенерация второй матрицы')
                    num1 = mtx_gnrt()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = subb(num, num1)
                    elif sposob == '2':
                        res = submx(num,num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '3':
                    print('\nГенерация матрицы')
                    num = mtx_gnrt()
                    kk = int(input('Введите число: '))
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = multch(num, kk)
                    elif sposob == '2':
                        res = multchmx(num,kk)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '4':
                    print('\nГенерация первой матрицы')
                    num = mtx_gnrt()
                    print('\nГенерация второй матрицы')
                    num1 = mtx_gnrt()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = mult(num, num1)
                    elif sposob == '2':
                        res = multmx(num, num1)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '5':
                    print('\nГенерация первой матрицы')
                    num = mtx_gnrt()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = trns(num)
                    elif sposob == '2':
                        res = trnsmx(num)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '6':  
                    print('\nГенерация первой матрицы')
                    num = mtx_gnrt()
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        v = time.time()
                        res = determinant(num,1)
                        if (time.time()-v)>0.0000001:
                            print(f"Время работы программы: {time.time()-v} секунд")
                        else:
                            print("Слишком быстро, начальника, мы не успеваем считать!")
                    elif sposob == '2':
                        print('Определитель = ',dettmx(num))
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '0':
                    break
                if cmd == '1' or cmd == '2' or cmd == '3' or cmd == '4' or cmd == '5':
                    #Проверка на ошибки
                    try:
                        if res != None:
                            print('\nПолученная матрица: ')
                            out(res)
                            zapis(res)
                        else:
                            print('Введите заново\n')
                    except ValueError:
                        print('\nПолученная матрица: ')
                        out(res)
                        zapis(res)
                elif cmd == '6':
                    print('Определитель = ',res,'\n')
                else:
                    print('Программа остановлена')
                    exit()
        elif inpt == '4':
            m1=[[1,2,3],
                [4,5,6],
                [7,8,9]]
            m2=[[9,8,7],
                [6,5,4],
                [3,2,1]]
            print('Тестовая матрица 1 = ')
            out(m1)
            print('Тестовая матрица 2 = ')
            out(m2)
            while(True):
                cmd = input('\nВыберите операцию:\n1)Сложение\n2)Вычитание\n3)Умножение на число\n4)Умножение\n5)Траспонирование\n6)Вычисление определителя\n0)Выбрать другой способ ввода\n\nВведите любой другой символ для выхода из программы\n>>>  ')
                if cmd == '1':
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = summ(m1, m2)
                    elif sposob == '2':
                        res = summx(m1,m2)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '2':
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = subb(m1, m2)
                    elif sposob == '2':
                        res = submx(m1,m2)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '3':
                    print('Преобразование матрицы m1')
                    kk = int(input('Введите число: '))
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = multch(m1, kk)
                    elif sposob == '2':
                        res = multchmx(m1,kk)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '4':
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = mult(m1, m2)
                    elif sposob == '2':
                        res = multmx(m1, m2)
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '5':
                    print('Преобразование матрицы m1')
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        res = trns(m1)
                    elif sposob == '2':
                        res = trnsmx(m1)
                    else:
                        print('Неправильная команда!!!')
                        break 
                elif cmd == '6':
                    print('Преобразование матрицы m1')
                    sposob = input('\nВыберите способ решения\n1)Своя ф-ция\n2)Стандартная библиотека\n')
                    if sposob == '1':
                        v = time.time()
                        res = determinant(m1,1)
                        if (time.time()-v)>0.0000001:
                            print(f"Время работы программы: {time.time()-v} секунд")
                        else:
                            print("Слишком быстро, начальника, мы не успеваем считать!")
                    elif sposob == '2':
                        print('Определитель = ',dettmx(m1))
                    else:
                        print('Неправильная команда!!!')
                        break
                elif cmd == '0':
                    break
                if cmd == '1' or cmd == '2' or cmd == '3' or cmd == '4' or cmd == '5':
                    #Проверка на ошибки
                    try:
                        if res != None:
                            print('\nПолученная матрица: ')
                            out(res)
                            zapis(res)
                        else:
                            print('Введите заново\n')
                    except ValueError:
                        print('\nПолученная матрица: ')
                        out(res)
                        zapis(res)
                elif cmd == '6':
                    print('Определитель = ',res,'\n')
                else:
                    print('Программа остановлена')
                    exit()
        elif inpt == '5':
            dop_zadanie()
            # график транспонирования 
            o1=sorted(o1)
            transnptime=sorted(transnptime)
            transtime=sorted(transtime)
            z = np.array(transnptime)
            g = np.array(o1)
            z1 = np.array(transtime)
            g1 = np.array(o1)
            plt.title('График транспонирования')
            plt.ylabel('Время работы')
            plt.xlabel('Количество элементов')
            plt.grid()
            plt.plot(g, z)
            plt.plot(g1, z1)
            plt.plot(g, z,label = 'numpy')
            plt.plot(g1, z1,label = 'Своя функция')
            plt.legend()
            plt.show()

            # график определителя
            o2=sorted(o2)
            detnptime=sorted(detnptime)
            dettime=sorted(dettime)
            h = np.array(detnptime)
            q = np.array(o2)
            h1 = np.array(dettime)
            q1 = np.array(o2)
            plt.title('График определителя')
            plt.ylabel('Время работы')
            plt.xlabel('Количество элементов')
            plt.grid()
            plt.plot(q, h,label = 'numpy')
            plt.plot(q1, h1,label = 'Своя функция')
            plt.legend()
            plt.show()
            print('Вывод: наши функции - неоптимизированное говно, numpy рулит')
        else:
            print('Программа остановлена')
            break


#-------------------- 3 --------------------#
def Minor(m,x,y):
    return [row[:y] + row[y+1:] for row in (m[:x]+m[x+1:])]
def inverse(num): #Вычисление обратной матрицы
    """[Вычисление обратной матрицы]

    Args:
        num ([list]): [Матрица]

    Returns:
        [list]: [Обратная матрица]
    """
    try:
        deet = determinant(num,1)
        cofactors = []
        for r in range(len(num)):
            cofactorRow = []
            for c in range(len(num)):
                minor = Minor(num,r,c)
                cofactorRow.append(((-1)**(r+c)) * determinant(minor,1))
            cofactors.append(cofactorRow)
        cofactors = trns(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c]/deet
        return cofactors
    except ZeroDivisionError:
        return(' ')

##### Функции Ввода 
def SLAU_mtx():
    """[Ввод матрицы СЛАУ]
    - Пользователь вводит ранг матрицы
    - Пользователь вводит значения матрицы поэлементно
        Returns:
            [list]: [Матрица СЛАУ]
    """
    def cock():  #Создание матрицы 1
        try:
            num[i][j] = float(input())
            return num[i][j]
        except ValueError:
            print('Ошибка ввода!!!\nВведите элемент заново\n')
            return cock()
        
    try:
        global x
        x = int(input('Введите ранг матрицы\n'))
        while x == 0:
            print('Матрицы не существует\n')
            x = int(input('Введите ранг матрицы отличный от нуля\n'))
        num = [[0 for n in range(x)] for nn in range(x)]
        print('Введите значения матрицы поэлементно')
        for i in range(x):
            for j in range(x):
                cock()
        print('Вы ввели матрицу:')
        out(num)
        return num 
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново ранг матрицы\n')
        return(SLAU_mtx())
def Coef():  #Ввод свободных членов 
    """[Ввод столбца свободных членов]
    -Пользователь вводит значения поэлементно
    Returns:
        [list]: [Массив свободных членов]
    """
    def cock1():  #Создание матрицы 1
        try:
            num1.append(float(input()))
            return num1
        except ValueError:
            print('Ошибка ввода!!!\nВведите элемент заново\n')
            return cock1()
    num1 = []
    print('Введите значения свободные члены')
    for i in range(x):
        cock1()
    print('Вы ввели:')
    print(num1,'\n') 
    return num1
def SLAU_gnrt(): #Ручная генерация матрицы
    """[Ручная генерация матрицы СЛАУ]
    - Пользователь вводит ранг матрицы
    - Пользователь вводит диапазон значений
    Returns:
        [list]: [Матрица СЛАУ]
    """
    mtx = []
    try:
        global x
        x = int(input('Введите ранг матрицы\n'))
        minn = float(input('Введите диапазон значений:\nВведите минимальное значение>>> ')) 
        maxx = float(input('Введите максимальное значение>>> '))
        for i in range(x):
            row = []
            for j in range(x):
                gen_num = round(uniform(minn,maxx),1)
                row.append(gen_num)
            mtx.append(row)
        print('Сгенирированная матрица = ')
        out(mtx)
        return mtx
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново\n')
        return SLAU_gnrt()
def Coef_gnrt():  #Генерация массива свободных членов
    """[Генерация массива свободных членов]
    - Пользователь вводит диапазон значений
    Returns:
        [list]: [Массив свободных членов]
    """
    mtx2 = []
    try:
        minn = float(input('Введите диапазон значений для столбца свободных коэф-тов :\nВведите минимальное значение>>> ')) 
        maxx = float(input('Введите максимальное значение>>> '))
        for i in range(x):
            gen_num = round(uniform(minn,maxx),1)
            mtx2.append(gen_num)
        print('Сгенирированные свободные коэф-ты = ')
        print(mtx2)
        return mtx2
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново\n')
        return Coef_gnrt()


#######################
# Условие завершения программы на основе вычисления
# расстояния между соответствующими элементами соседних
# итераций в методе решения 
def isNeedToComplete(x_old, x_new):

    eps = 0.0001
    sum_up = 0
    sum_low = 0
    for k in range(0, len(x_old)):
        sum_up += ( x_new[k] - x_old[k] ) ** 2
        sum_low += ( x_new[k] ) ** 2
        
    return math.sqrt( sum_up / sum_low ) < eps
# Процедуры решения
def yacoby(a1):
    """[Метод Якоби для решения СЛАУ]

    Args:
        a1 ([list]): [Сшитая матрица(Матрица СЛАУ + столбец коэф.)]

    Returns:
        [list]: [Решение]
    """
    b =[]
    a = copy.deepcopy(a1)
    for i in range(len(a)):
        b.append(a[i][-1])
    for i in range(len(a)):
        a[i] = a[i][:-1]
    count = len(b) # количество корней
    x = [1 for k in range(0, count) ] # начальное приближение корней  
    numberOfIter = 0  # подсчет количества итераций
    MAX_ITER = 100    # максимально допустимое число итераций
    while( numberOfIter < MAX_ITER ):
        x_prev = copy.deepcopy(x)
        for k in range(0, count):
            S = 0
            for j in range(0, count):
                if( j != k ): S = S + a[k][j] * x[j] 
            x[k] = b[k]/a[k][k] - S / a[k][k]
        if isNeedToComplete(x_prev, x) : # проверка на выход
            break
        numberOfIter += 1
    if numberOfIter == 100:
        return('Матрица расходится')
    print('Количество итераций на решение: ', numberOfIter)  
    return x
def GousJ1(num): #Приводит матрицу к единичной форме 
    for i in range(len(num)):
        for j in range(len(num[i])):
            if i == j:
                num[i] = [num[i][k]/num[i][i] for k in range(len(num)+1)]
    return(num)
def GousJ(num): # Основной алгоритм поиска решения методом Жордана-Гауса
    """[Метод Жордана-Гауса для решения СЛАУ]

    Args:
        num ([list]): [Сшитая матрица(Матрица СЛАУ + столбец коэф.)]

    Returns:
        [list]: [Решение]
    """
    for i in range(len(num)):
        num = changeZero(num)
        GousJ1(num)
        for j in range (len(num)):
            if i!=j:
                num[j] = [num[j][k] - num[i][k]*num[j][i] for k in range(len(num)+1)]
    for i in range(len(num)):
        num[i] = num[i][-1]
    return num
def GousJFR(num): #Жордан-Гаус с типом Fraction
    """[Метод Жордана-Гауса для решения СЛАУ с типом Fraction]

    Args:
        num ([list]): [Сшитая матрица(Матрица СЛАУ + столбец коэф.)]

    Returns:
        [list]: [Решение в типе Fraction]
    """
    for i in range(len(num)):
        num[i] = [str(num[i][j]) for j in range(len(num)+1)]
        num[i] = [Fraction(num[i][j]) for j in range(len(num)+1)]
    for i in range(len(num)):
        num = changeZero(num)
        GousJ1(num)
        for j in range (len(num)):
            if i!=j:
                num[j] = [num[j][k] - num[i][k]*num[j][i] for k in range(len(num)+1)]
    for i in range(len(num)):
        num[i] = num[i][-1]
    return num

### Проверки
def ErrorZero(num):
    for i in range(len(num)):
        if num[i] ==[0]*len(num):
          return False
    err = trns(num)
    for i in range(len(num)):
        if err[i] ==[0]*len(num):
          return False
    return True
def changeZero(num2):
    num = copy.deepcopy(num2)
    for i in range(len(num)):
        if num[i][i] == 0.0:
            for j in range(len(num)):
                if j!=i and num[j][i]!=0:
                    num[i] = [num[i][k]+num[j][k] for k in range(len(num)+1)]
                    break
    return num
def proverka(num,num1):
    """[Проверка и сшивка СЛАУ]
    -Проверяет возможно ли решить СЛАУ
    -Сшивает матрицу(добавляет столбец коэф-в с матрицу СЛАУ)

    Args:
        num ([list]): [Матрица СЛАУ]
        num1 ([list]): [Массив свободных Коэф-в]

    Returns:
        [list]: [Сшитая СЛАУ]
    """
    if ErrorZero(num):
        for i in range(len(num)):
            num[i].append(num1[i])
        num = changeZero(num)
        return num
    return 'СЛАУ не решаема'
         
#--- ОСНОВНОЙ КОД РЕШЕНИЯ СЛАУ ---#
def SLAU():
    """[Запуск программы по решению СЛАУ]

    """
    while(True):
        inpt = input('\nВыберите метод ввода СЛАУ:\n1)Ввод вручную\n2)Импорт CSV файла\n3)Случайная генерация\n\nВведите любой другой символ для вывхода из программы\n')
        if inpt =='1':
            a = SLAU_mtx()
            cord = LA.cond(a)
            b = Coef()
            a = proverka(a,b)
            if type(a) == type('War... War never changes.'):
                print(a)
                return SLAU()
            a_prev = copy.deepcopy(a)
            print('Обратная матрица: ')
            out(inverse(a))
            if type(inverse(a)) == type('War... War never changes.'):
                print('Матрица вырожденная\nОбратная матрица не существует\n')
                return SLAU()
            print( 'Решение методом Якоби: ', yacoby(a))
            print('Число обусловленности = ',cord,'\n')
            if cord > 100 or type(yacoby(a)) == type('War... War never changes.'):
                print('Решение методом Гаусса-Жордана: ', GousJ(a_prev))
                print('Решение методом Гаусса-Жордана (правильные дроби): ')
                out(GousJFR(a))
        elif inpt =='2':
            print('Введите название первого файла: ')
            a = CSV()
            try:
                for i in range(len(a)):
                    for j in range(len(a[i])):
                        a[i][j] = float(a[i][j])
            except ValueError:
                print('Постороннее значение в файле\nДальнейшее решение невозможно')
                return SLAU()
            cord = LA.cond(a)
            print('Введите название второго файла: ')
            b = CSV()
            b = b[0]
            try:
                for i in range(len(b)):
                    b[i] = float(b[i].replace(',','.'))
            except ValueError:
                print('Постороннее значение в файле\nДальнейшее решение невозможно')
                return SLAU()
            a = proverka(a,b)
            if type(a) == type('War... War never changes.'):
                print(a)
                return SLAU()
            a_prev = copy.deepcopy(a)
            print('Обратная матрица: ')
            out(inverse(a))
            if type(inverse(a)) == type('War... War never changes.'):
                print('Матрица вырожденная\nОбратная матрица не существует\n')
                return SLAU()
            print( 'Решение методом Якоби: ', yacoby(a) )
            print('Число обусловленности = ',cord,'\n') 
            if cord > 100 or type(yacoby(a)) == type('War... War never changes.'):
                print('Решение методом Гаусса-Жордана: ', GousJ(a_prev))
                print('Решение методом Гаусса-Жордана (правильные дроби): ')
                out(GousJFR(a))
        elif inpt =='3':
            a = SLAU_gnrt()
            cord = LA.cond(a)
            b = Coef_gnrt()
            a = proverka(a,b)
            if type(a) == type('War... War never changes.'):
                print(a)
                return SLAU()
            a_prev = copy.deepcopy(a)
            print('Обратная матрица: ')
            out(inverse(a))
            if type(inverse(a)) == type('War... War never changes.'):
                print('Матрица вырожденная\nОбратная матрица не существует\n')
                return SLAU()
            print( 'Решение методом Якоби: ', yacoby(a) )
            print('Число обусловленности = ',cord,'\n') 
            if cord > 100 or type(yacoby(a)) == type('War... War never changes.'):
                print('Решение методом Гаусса-Жордана: ', GousJ(a_prev))
                print('Решение методом Гаусса-Жордана (правильные дроби): ')
                out(GousJFR(a))
        else:
            break


#-------------------- 4 --------------------#
def int_lagr(a,b): #Интерполяция Лагранжем
    """[Интерполяция Лагранжем]

    Args:
        a ([list]): [Массив значений по X]
        b ([list]): [Массив значений по Y]

    Returns:
        [list]: [Решение]
    """
    yn=0
    YN=[]
    N=len(a)
    for k in range(len(a)):
        z=a[k]
        yn=0
        for i in range (0,len(a)):# цикл на вычисление многочлена лагранжа
            L=1
            for j in range (0,len(a)):
                if i != j:
                    try:
                        L*=(z-a[j])/(a[i]-a[j])
                    except ZeroDivisionError:
                        L*=1
            yn+=b[i]*L
        YN.append(yn)
    return YN
def int_stnd(a,b): #Интерполяция Ньютоном
    """[Интерполяция Ньютоном]
    
    Args:
        a ([list]): [Массив значений по X]
        b ([list]): [Массив значений по Y]

    Returns:
        [list]: [Решение]
    """
    np_y = list(np.interp(a, a, b))
    print(a)
    print(b)
    print(np_y)
    return np_y  
def aprx_lin(a,b):  # аппроксимация линейной функцией
    """[Аппроксимация линейной функцией]
    
    Args:
        a ([list]): [Массив значений по X]
        b ([list]): [Массив значений по Y]

    Returns:
        [list]: [Решение]
    """
    x0=a[0]
    x1=a[-1]
    step= 0.1 

    a1 = copy.deepcopy(a)
    b1 = copy.deepcopy(b)
    Linear_massive_x = [ ]
    Linear_massive_y = [ ]
    xx = x0
    for z in range(int((x1 - x0) / step) + 1):
        s1 = 0; s2 = 0; s3 = 0; s4 = 0
        yy = 0
        c1 = 1; c0 = 1
        for i in range(len(a)):
            s1 += a1[ i ] * a1[ i ]
            s2 += a1[ i ]
            s3 += a1[ i ] * b1[ i ]
            s4 += b1[ i ]
        try:
            c1 = (s3 * len(a) - s2 * s4) / (s1 * len(a) - s2 * s2)
        except ZeroDivisionError:
            c1 = 0
        try:
            c0 = (s1 * s4 - s2 * s3) / (s1 * len(a) - s2 * s2)
        except ZeroDivisionError:
            c0 = 0
        yy = c0 + c1 * xx
        #print('При x = ', xx, 'значение в точке = ', yy)
        Linear_massive_x.append(xx)
        Linear_massive_y.append(yy)
        xx = xx + step 
    return Linear_massive_x, Linear_massive_y
def aprx_quad(a,b):  #Аппроксимация квадратичной функцией
    """[Аппроксимация квадратичной функцией]
    
    Args:
        a ([list]): [Массив значений по X]
        b ([list]): [Массив значений по Y]

    Returns:
        [list]: [Решение]
    """
    x0=a[0]
    x1=a[-1]
    step= 0.1 

    list_x = copy.deepcopy(a)
    list_y = copy.deepcopy(b)
    Quadrannntic_massive_x = [ ]
    Quadrannntic_massive_y = [ ]
    xx = x0
    for z in range(int((x1 - x0) / step) + 1):
        az = [ [ ], [ ], [ ] ]
        bz = [ ]
        s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0; s6 = 0; s7 = 0;
        for i in range(len(list_x)):
            s1 += list_x[ i ] ** 4
            s2 += list_x[ i ] ** 3
            s3 += list_x[ i ] ** 2
            s4 += list_x[ i ]
            s5 += (list_x[ i ] ** 2) * list_y[ i ]
            s6 += list_x[ i ] * list_y[ i ]
            s7 += list_y[ i ]
        az[ 0 ] += [ s1, s2, s3 ]
        az[ 1 ] += [ s2, s3, s4 ]
        az[ 2 ] += [ s3, s4, len(list_x) ]
        bz += [ s5, s6, s7 ]
        #!!!!
        azz = proverka(az,bz)
        if type(azz) == type('War... War never changes.'):
                print(azz)
        a_1 = GousJ(azz)
        yy = a_1[ 2 ] + a_1[ 1 ] * xx + a_1[ 0 ] * (xx ** 2)
        Quadrannntic_massive_x.append(xx)
        Quadrannntic_massive_y.append(yy)
        xx += step
    y0diff=0
    for i in range(len(b)-1):
        y0diff+=(b[i]- Quadrannntic_massive_y[i])**2
    print("дисперсия равна",y0diff)
    return Quadrannntic_massive_x,Quadrannntic_massive_y
def aprx_nrmDist(a,b): #Аппроксимация норм. Распределения
    """[Аппроксимация норм. Распределения]
    
    Args:
        a ([list]): [Массив значений по X]
        b ([list]): [Массив значений по Y]

    Returns:
        [list]: [Решение]
    """
    x0=a[0]
    x1=a[-1]
    step= 0.1 
    list_x = copy.deepcopy(a)
    list_y = copy.deepcopy(b)
    normal_massive_x = []
    normal_massive_y = []
    #x0=x_list[0]
    x1=list_x[-1]
    xx = x0
    x0, x1, step = x0, x1, step
    a = [[],[],[]]
    b = []
    s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0; s6 = 0; s7 = 0;
    for i in range(len(list_x)):
        s1 = s1 + list_x[i]**4
        s2 = s2 + list_x[i]**3
        s3 = s3 + list_x[i]**2
        s4 = s4 + list_x[i]
        s5 = s5 + (list_x[i]**2)*list_y[i]
        s6 = s6 + list_x[i]*list_y[i]
        s7 = s7 + list_y[i]
    a[0].append(s1); a[0].append(s2); a[0].append(s3)
    a[1].append(s2); a[1].append(s3); a[1].append(s4)
    a[2].append(s3); a[2].append(s4); a[2].append(len(list_x))
    b.append(s5); b.append(s6); b.append(s7)
    #!!!!
    a = proverka(a,b)
    if type(a) == type('War... War never changes.'):
            print(a)
    a_1 = GousJ(a)
    a_coef = math.e**(a_1[0]-((a_1[1]**2)/4*a_1[2]))
    b_coef = -1/(a_1[2])
    c_coef = (-a_1[1])/(2*a_1[2])
    for z in range(0, int((x1-x0)/step)+1):
        try:
            yy = a_coef*math.e**(-(((xx-c_coef)**2)/b_coef**2))
        except ZeroDivisionError:
            yy = 0
        normal_massive_x.append(xx)
        normal_massive_y.append(yy)
        xx = xx + step
    return normal_massive_x,normal_massive_y

def CSV_interpol(): # Считываем из CSV файла
    try:
        with open(input('Введите в формате: "Название"."csv"\n'),newline='') as csvfile:
            reader = csv.reader(csvfile,delimiter=';')
            lst=[]
            lst1 = []
            for row in reader:
                lst.append(row[0])
                lst1.append(row[1])
        return [lst,lst1]
    except FileNotFoundError:
        print('Искомый файл не найден!!!\n')
        return CSV_interpol()

#--- ОСНОВНОЙ КОД РЕШЕНИЯ Интерполяции ---#
def Int_Aprx():
    """[Запуск программы по работе с Интерполяцией/Апроксимацией]

    """
    while(True):
        inpt = input('\nВыберите метод ввода :\n1)Импорт CSV файла\n2)Тестовые значения\n\nВведите любой другой символ для вывхода из программы\n')
        if inpt =='1':
            kk = CSV_interpol()
            try:
                for i in range(len(kk)):
                    for j in range(len(kk[i])):
                        kk[i][j] = float(kk[i][j])
            except ValueError:
                print('Постороннее значение в файле\nДальнейшее решение невозможно')
                return Int_Aprx()
            x = kk[:-1]
            x = sum(x, start=[])
            y = kk[-1:]
            y = sum(y, start=[])
            print(x)
            print(y)
            
            plt.figure(figsize = (15, 8))
            one = int_lagr(x,y)
            plt.subplot(2, 3, 1)
            plt.plot(x, y, 'bo')
            xnew=np.linspace(np.min(x),np.max(x),100)
            spl=make_interp_spline(x, y, k=3)
            power_smooth=spl(xnew)
            plt.plot(xnew, power_smooth)
            plt.plot(x,one,'o')
            plt.title('График интерполяции Лагранжем')

            two = int_stnd(x,y)
            plt.subplot(2, 3, 2)
            plt.plot(x, y, 'bo')
            xnew=np.linspace(np.min(x),np.max(x),100)
            spl=make_interp_spline(x, y, k=3)
            power_smooth=spl(xnew)
            plt.plot(xnew, power_smooth)
            plt.plot(x,two,'o')
            plt.title('График интерполяцией numpy')

            three = aprx_lin(x,y)
            plt.subplot(2, 3, 4)
            #plt.plot(a, b, 'bo')
            plt.plot(three[:-1], three[-1:],'bo')
            plt.title('График линейной аппроксимации') 

            four = aprx_quad(x,y)
            plt.subplot(2, 3, 5)
            #plt.plot(a, b, 'bo')
            plt.plot(four[:-1], four[-1:],'bo')
            plt.title('График квадратичной аппроксимации')
            #A[1]

            five = aprx_nrmDist(x,y)
            plt.subplot(2, 3, 6)
            plt.plot(five[:-1], five[-1:], 'bo')
            plt.title('График  апроксимации нормального распределения')
            plt.tight_layout()
            plt.show()

        elif inpt =='2':
            x=[1,2,3,4,5,6,7,8,9,10]
            y=[1,3,2,5,4,7,3,8,4,6]
            x0=x[0]
            x1=x[-1]
            step= 0.1
            print('X = ',x)
            print('Y = ',y)
            print('Шаг = ',step)

            plt.figure(figsize = (15, 8))
            one = int_lagr(x,y)
            plt.subplot(2, 3, 1)
            plt.plot(x, y, 'bo')
            xnew=np.linspace(np.min(x),np.max(x),100)
            spl=make_interp_spline(x, y, k=3)
            power_smooth=spl(xnew)
            plt.plot(xnew, power_smooth)
            plt.plot(x,one,'o')
            plt.title('График интерполяции Лагранжем')

            two = int_stnd(x,y)
            plt.subplot(2, 3, 2)
            plt.plot(x, y, 'bo')
            xnew=np.linspace(np.min(x),np.max(x),100)
            spl=make_interp_spline(x, y, k=3)
            power_smooth=spl(xnew)
            plt.plot(xnew, power_smooth)
            plt.plot(x,two,'o')
            plt.title('График интерполяцией numpy')

            three = aprx_lin(x,y)
            plt.subplot(2, 3, 4)
            #plt.plot(a, b, 'bo')
            plt.plot(three[:-1], three[-1:],'bo')
            plt.title('График линейной аппроксимации') 

            four = aprx_quad(x,y)
            plt.subplot(2, 3, 5)
            #plt.plot(a, b, 'bo')
            plt.plot(four[:-1], four[-1:],'bo')
            plt.title('График квадратичной аппроксимации')
            #A[1]

            five = aprx_nrmDist(x,y)
            plt.subplot(2, 3, 6)
            plt.plot(five[:-1], five[-1:], 'bo')
            plt.title('График  апроксимации нормального распределения')
            plt.tight_layout()
            plt.show()


#-------------------- 5 --------------------#
def CSV_vvet(nazv,sep):
    with open(nazv,newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=sep)
        el=[]
        for row in reader:
            el.append(list(row))
    return el
#--- ОСНОВНОЙ КОД РЕШЕНИЯ Вейвлет ---#
def velvet():
    """[Запуск программы по работе с Вейвлет/Фурье Анализом]
    - Запускает интерфейс программы
    - все функции внутри программы работают через функции библиотек python
    """
    print('Тестовые файлы из ТЗ: \n 1) Test_sin.csv \n 2) Line.csv \n 3) sin_peak.csv \n 4) 2_sin_5_4.5.csv')
    File = str(input('Введите название файла: '))
    sep = str(input('Укажитель разделитель (, или ;): '))
    l=CSV_vvet(File,sep)
    for i in range(len(l)):
        l[i][0]=float(l[i][0])
        l[i][1]=float(l[i][1])
        print(f'Значение {i+1}: [{l[i][0]},{l[i][1]}]')


    x = [i[0] for i in l]
    y = [i[1] for i in l]
    fi = list(fft(np.array(y))) #Прямое Фурье преобраз

    rcParams['figure.figsize'] = (10, 10)
    rcParams['figure.dpi'] = 70

    plt.subplot(2, 1, 1)

    plt.plot(x,y,color='blue',lw=2)
    plt.grid(b=True, color='blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Изначальный сигнал. Файл {File}.',fontsize=20, loc = 'right')
    plt.tick_params(labelsize = 30)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)

    plt.subplot(2, 1, 2)

    plt.plot(x,fi,color='blue',lw=2)
    plt.grid(b=True, color='blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'FFT. Файл {File}.',fontsize=20, loc = 'right')
    plt.tick_params(labelsize = 30)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)
    plt.show()


    ###############################
    #Daubechies (db)
    x = [i[0] for i in l]
    y = [i[1] for i in l]
    lvl = pywt.wavedec(y, 'db6', level=4) #точки, db6 -метод сколько уровней - level

    rcParams['figure.figsize'] = (10, 10)
    rcParams['figure.dpi'] = 70

    # Уровень 1
    plt.subplot(4, 1, 1)
    plt.plot(lvl[-4],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Daubechies Уровень 1 {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)

    # Уровень 2
    plt.subplot(4, 1, 2)
    plt.plot(lvl[-3],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Daubechies Уровень 2 {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)

    # Уровень 3
    plt.subplot(4, 1, 3)
    plt.plot(lvl[-2],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Daubechies Уровень 3 {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)

    # Уровень 4
    plt.subplot(4, 1, 4)
    plt.plot(lvl[-1],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Daubechies Уровень 4 {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)
    plt.show()
    ###########
    #Haar (haar)

    x = [i[0] for i in l]
    y = [i[1] for i in l]
    lvl = pywt.wavedec(y, 'haar', level=4)

    rcParams['figure.figsize'] = (10, 10)
    rcParams['figure.dpi'] = 70

    plt.subplot(4, 1, 1)

    plt.plot(lvl[-4],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Haar Уровень 1  {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)



    plt.subplot(4, 1, 2)

    plt.plot(lvl[-3],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Haar Уровень 2 {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)


    plt.subplot(4, 1, 3)

    plt.plot(lvl[-2],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Haar Уровень 3 {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)


    plt.subplot(4, 1, 4)

    plt.plot(lvl[-1],color='blue',lw=2)
    plt.grid(b=True, color='Blue', alpha=0.75, linestyle=':', linewidth=1)
    plt.title(f'Haar Уровень 4 {File}.',fontsize=10, loc = 'right')
    plt.tick_params(labelsize = 20)
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)
    plt.show()


#-------------------- 6 --------------------#
### Ввод ОДУ
def vvod(): #Ввод ОДУ
    try:
        v_dec = input('\nВведите уравнение >>> ')
        vod = eval(v_dec.lower())
        try:
            f = lambdify([x,y],vod)
            return f
        except KeyError:
            print('Делить на 0 нельзя. а-я-я-й')
            return vvod()
    except (NameError,SyntaxError):
        print('ОШИБКА!!! Введены посторонние символы\n Введите заново')
        return vvod()
def vvod1(): #Ввод системы ОДУ
    try:
        v_dec = input('\nВведите уравнение 1 >>> ')
        vod = eval(v_dec.lower())
        v_dec1 = input('Введите уравнение 2 >>> ')
        vod1 = eval(v_dec1.lower())
        try:
            f = lambdify([x,y,z],vod)
            f1 = lambdify([x,y,z],vod1)
            return [f, f1]
        except KeyError:
            print('Делить на 0 нельзя. а-я-я-й')
            return vvod1()
    except (NameError,SyntaxError):
        print('ОШИБКА!!! Введены посторонние символы\n Введите заного')
        return vvod1()
    
### Вывод ОДУ
def vivod(n, func): #Вывод ОДУ
    m1 = []
    m2 = []
    for i in range(n):
        m1.append(func[i][1])
        m2.append(func[i][2])
    out(func)
    return [m1,m2]
def vivodSys(n, func): # Вывод системы ОДУ
    m1 = []
    m2 = []
    m3 = []
    for i in range(n):
        m1.append(func[i][1])
        m2.append(func[i][2])
        m3.append(func[i][3])
    out(func)
    return [m1,m2,m3]

### Методы решения
def AYEler(F,x0,y0,n,step): #Решение ОДУ методом Эйлера-коши
    res =[]
    for i in range(n):
        y1 = y0 + step*F(x0,y0)
        x1 = x0 + step
        y2 = y0 + step/2*(F(x0,y0)+F(x1,y1))
        res.append([i,x1,y2])
        x0 = x1
        y0 = y2
    return res
def RNGkuti(F,x0,y0,n,step): #Решение ОДУ методом Рунге-Кутты
    res1 = []
    for i in range(n):
        k1 = step*F(x0,y0)
        k2 = step*F(x0 + step/2,y0 + k1/2)
        k3 = step*F(x0 + step/2,y0 + k2/2)
        k4 = step*F(x0 + step,y0 + k3)
        y1 = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        x1 = x0 + step
        res1.append([i,x1,y1])
        x0 = x1
        y0 = y1
    return res1   
def AYElerSys(F,x0,y0,z0,n,step): #Решение системы ОДУ методом Эйлера-коши
    res =[]
    for i in range(n):
        y1 = y0 + step*F[0](x0,y0,z0)
        z1 = z0 + step*F[1](x0,y0,z0)
        x1 = x0 + step
        y2 = y0 + step/2*(F[0](x0,y0,z0)+F[0](x1,y1,z1))
        z2 = z0 + step/2*(F[1](x0,y0,z0)+F[1](x1,y1,z1))
        res.append([i,x1,y2,z2])
        x0 = x1
        y0 = y2
        z0 = z2
    return res
def RNGkutiSys(F,x0,y0,z0,n,step): #Решение системы ОДУ методом Рунге-Кутты
    res1 = []
    for i in range(n):
        k1 = step*F[0](x0,y0,z0)
        k2 = step*F[0](x0 + step/2,y0 + k1/2,z0 + k1/2)
        k3 = step*F[0](x0 + step/2,y0 + k2/2,z0 + k2/2)
        k4 = step*F[0](x0 + step,y0 + k3,z0 +k3)

        l1 = step*F[1](x0,y0,z0)
        l2 = step*F[1](x0 + step/2,y0 + k1/2,z0 + k1/2)
        l3 = step*F[1](x0 + step/2,y0 + k2/2,z0 + k2/2)
        l4 = step*F[1](x0 + step,y0 + k3,z0 +k3)

        y1 = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        z1 = z0 + 1/6*(l1 + 2*l2 + 2*l3 + l4)
        x1 = x0 + step
        res1.append([i,x1,y1,z1])
        x0 = x1
        y0 = y1
        z0 = z1
    return res1

def decor():
    try:
        m_dec = [float(input('Введите x0 >>> ')),float(input('Введите y0 >>> ')),int(input('Введите начальную точку интервала >>> ')),int(input('Введите конечную точку интервала >>> ')),int(input('Введите кол-во точек >>> '))]
        return m_dec
    except ValueError:
        print('ОШИБКА!!! Введены посторонние символы\n Введите заново')
        return decor()

def y_sh(kk,ys,step): # y' для ОДУ
    for i in range(len(ys)-1):
        y_1 = (ys[i+1]-ys[i])/step
        kk[i].append(y_1)

    out(kk)
    kk = kk[:-1]
    kk = [[kk[i][1],kk[i][3]-kk[i][2]] for i in range(len(kk))]
    return kk
def y_shSys(kk,ys,zs,step): # y' для системы ОДУ
    for i in range(len(ys)-1):
        y_1 = (ys[i+1]-ys[i])/step
        z_1 = (zs[i+1]-zs[i])/step
        kk[i].append(y_1)
        kk[i].append(z_1)


    out(kk)
    kk = kk[:-1]
    kk = [[kk[i][1],kk[i][4]-kk[i][2],kk[i][5]-kk[i][3]] for i in range(len(kk))]
    return kk

def tst(): #Тестовые задания
    a = 0
    b = 1
    n = int(input('Введите кол-во точек >>> '))
    try:
        step = (b - a)/n
    except ZeroDivisionError:
        print('\nРешение невозможно. Кол-во шагов = 0')
        return tst()
    #Вариант 2
    print(' Вариант 2 '.center(45,'-'))
    F = lambdify([x,y], eval('cos(x+y)'))
    x0 = 0
    y0 = 0.4
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title('cos(x+y)\ny(0) = 0.4')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>>')

    #Вариант 3
    print(' Вариант 3 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('x*y+z')),lambdify([x,y,z],eval('y-z'))]
    x0 = 0
    y0 = 1
    z0 = 0
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title('x*y+z\ny-z\n y(0) = 1, z(0) = 0')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>>')

    #Вариант 4
    print(' Вариант 4 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('x**2+z')),lambdify([x,y,z],eval('y-z'))]
    x0 = 0
    y0 = 1
    z0 = 0
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title('x**2+z\ny-z\n y(0) = 1, z(0) = 0')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 5
    print(' Вариант 5 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('z**2+x')),lambdify([x,y,z],eval('x*y'))]
    x0 = 0
    y0 = 1
    z0 = -0.5
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title('z**2+x\nx*y\n y(0) = 1, z(0) = -0.5')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 6
    print(' Вариант 6 '.center(45,'-'))
    F = lambdify([x,y], eval('exp(-x)-y'))
    x0 = 0
    y0 = 1
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title('exp(-x)-y\ny(0) = 1')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 7
    print(' Вариант 7 '.center(45,'-'))
    F = lambdify([x,y], eval('sqrt(x)+y'))
    x0 = 0
    y0 = 1
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title('sqrt(x)+y\ny(0) = 1')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 8
    print(' Вариант 8 '.center(45,'-'))
    F = lambdify([x,y], eval('y*sin(x)+x'))
    x0 = 0
    y0 = 0.2
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title('y*sin(x)+x\ny(0) = 0.2')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 9
    print(' Вариант 9 '.center(45,'-'))
    F = lambdify([x,y], eval('y*cos(x)+x'))
    x0 = 0
    y0 = 0.1
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title('y*cos(x)+x\ny(0) = 0.1')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 10
    print(' Вариант 10 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('x+y+z')),lambdify([x,y,z],eval('y-z'))]
    x0 = 0
    y0 = 1
    z0 = -1
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title('x+y+z\ny-z\n y(0) = 1, z(0) = -1')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 11
    print(' Вариант 11 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('x*y+z')),lambdify([x,y,z],eval('y+x*z'))]
    x0 = 0
    y0 = 0
    z0 = 0.5
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title('x*y+z\ny+x*z\n y(0) = 0, z(0) = -0.5')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 12
    print(' Вариант 12 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('x**2-z')),lambdify([x,y,z],eval('y+x'))]
    x0 = 0
    y0 = 1
    z0 = 1
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'x*y+z\ny+x*z\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 13
    print(' Вариант 13 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('y-z')),lambdify([x,y,z],eval('y*z'))]
    x0 = 0
    y0 = 0.5
    z0 = 0
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'x*y+z\ny+x*z\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 14
    print(' Вариант 14 '.center(45,'-'))
    F = lambdify([x,y], eval('2*y-3*x**2-2'))
    x0 = 0
    y0 = 2
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'2*y-3*x**2-2\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 15
    print(' Вариант 15 '.center(45,'-'))
    F = lambdify([x,y], eval('x+y*sqrt(x)'))
    x0 = 0
    y0 = 0.2
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'2*y-3*x**2-2\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 16
    print(' Вариант 16 '.center(45,'-'))
    F = lambdify([x,y], eval('1+0.2*y*sin(x)-y**2'))
    x0 = 0
    y0 = 0.2
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'1+0.2*y*sin(x)-y**2\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 17
    print(' Вариант 17 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('-x*z')),lambdify([x,y,z],eval('y/x'))]
    x0 = 0
    y0 = 1
    z0 = 1
    try:
        func = AYElerSys(F,x0,y0,z0,n,step)
        func1 = RNGkutiSys(F,x0,y0,z0,n,step)
        print(' Решение методом Эйлера-Коши '.center(45,'-'))
        kk = vivodSys(n, func)
        xs,ys,zs = kk[0],kk[1],kk[2]
        print(' Решение методом Рунге Кутты '.center(45,'-'))
        kk1 = vivodSys(n, func1)
        xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

        fig = plt.figure(figsize = (7,7))
        plt.title(f'-x*z\ny/x\n y({x0}) = {y0}, z({x0}) = {z0}')
        ax = fig.add_subplot(projection='3d')
        ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
        ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
        ax.legend()
        fig.show()
        input('Введите любой символ чтобы продолжить >>> ')
    except ZeroDivisionError:
                print('ОШИБКА!!! обнаружено деление на 0')

    #Вариант 18
    print(' Вариант 18 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('(y+z)*x')),lambdify([x,y,z],eval('(-y+z)*x'))]
    x0 = 0
    y0 = 1
    z0 = 1
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'(y+z)*x\n(-y+z)*x\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 19
    print(' Вариант 19 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('-y*z+cos(x)/x')),lambdify([x,y,z],eval('-z**2+2.5*x/(1+x**2)'))]
    x0 = 0
    y0 = 0
    z0 = -0.2
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'-y*z+cos(x)/x\n-z**2+2.5x/(1+x**2)\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 20
    print(' Вариант 20 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('z-(2*y+0.25*z)*y')),lambdify([x,y,z],eval('exp(y)-(2+2*z)*y'))]
    x0 = 0
    y0 = 0.5
    z0 = 0.5
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'z-(2*y+0.25*z)*y\nexp(y)-(2+2*z)*y\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 21
    print(' Вариант 21 '.center(45,'-'))
    F = lambdify([x,y], eval('x*ln(y)+y*ln(x)'))
    x0 = 1
    y0 = 1
    step21 = (6 - 1)/n
    func = AYEler(F,x0,y0,n,step21)
    func1 = RNGkuti(F,x0,y0,n,step21)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'x*ln(y)+y*ln(x)\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 22
    print(' Вариант 22 '.center(45,'-'))
    F = lambdify([x,y], eval('exp(x)-y'))
    x0 = 0
    y0 = 0
    step22 = (2 - 0)/n
    func = AYEler(F,x0,y0,n,step22)
    func1 = RNGkuti(F,x0,y0,n,step22)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'exp(x)-y\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 23
    print(' Вариант 23 '.center(45,'-'))
    F = lambdify([x,y], eval('sqrt(x)+sqrt(y)'))
    x0 = 1
    y0 = 0.5
    step23 = (2 - 1)/n
    func = AYEler(F,x0,y0,n,step23)
    func1 = RNGkuti(F,x0,y0,n,step23)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'sqrt(x)+sqrt(y)\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 24
    print(' Вариант 24 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('z+0.5')),lambdify([x,y,z],eval('y-x'))]
    x0 = 0
    y0 = 0.5
    z0 = 0.5
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'z+0.5\ny-x\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 25
    print(' Вариант 25 '.center(45,'-'))
    F = lambdify([x,y], eval('y*sin(x)-y**2'))
    x0 = 1
    y0 = 0.5
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'y*sin(x)-y**2\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 26
    print(' Вариант 26 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('cos(y+2*z)+2')),lambdify([x,y,z],eval('2/(x+6*y**2)+x+1'))]
    x0 = 0
    y0 = 0.1
    z0 = 0.5
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'cos(y+2*z)+2\n2/(x+6*y**2)+x+1\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 27
    print(' Вариант 27 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('sin(x**2)+y+z')),lambdify([x,y,z],eval('x+y-z**2+1'))]
    x0 = 0
    y0 = 0.5
    z0 = 1
    
    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'sin(x**2)+y+z\nx+y-z**2+1\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 28
    print(' Вариант 28 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('ln(2*x+z)')),lambdify([x,y,z],eval('sqrt(4*x**2+y**2)'))]
    x0 = 0
    y0 = 1
    z0 = 1
    step28 = (4 - 0)/n

    func = AYElerSys(F,x0,y0,z0,n,step28)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step28)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'ln(2*x+z)\nsqrt(4*x**2+y**2)\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 29
    print(' Вариант 29 '.center(45,'-'))
    F = lambdify([x,y], eval('cos(x)/(1+y**2)'))
    x0 = 0
    y0 = 0
    step29 = (4 - 0)/n
    func = AYEler(F,x0,y0,n,step29)
    func1 = RNGkuti(F,x0,y0,n,step29)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'cos(x)/(1+y**2)\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 30
    print(' Вариант 30 '.center(45,'-'))
    F = lambdify([x,y], eval('exp(-x)*(y**2+1.04)'))
    x0 = 0
    y0 = 0
    func = AYEler(F,x0,y0,n,step)
    func1 = RNGkuti(F,x0,y0,n,step)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'exp(-x)*(y**2+1.04)\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 31
    print(' Вариант 31 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('cos(y+2*z)+4')),lambdify([x,y,z],eval('2/(x+4*y)+x+1'))]
    x0 = 0
    y0 = 0.1
    z0 = 0.5

    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'cos(y+2*z)+4\n2/(x+4*y)+x+1\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 32
    print(' Вариант 32 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('sqrt(x**2+2*y**2)+z')),lambdify([x,y,z],eval('cos(2*z)+x'))]
    x0 = 0
    y0 = 0.4
    z0 = 0.4

    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'sqrt(x**2+2*y**2)+z\ncos(2*z)+x\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 33
    print(' Вариант 33 '.center(45,'-'))
    F = [lambdify([x,y,z],eval('exp(-(y+z))+2*x')),lambdify([x,y,z],eval('x**2+y'))]
    x0 = 0
    y0 = 1
    z0 = 1

    func = AYElerSys(F,x0,y0,z0,n,step)
    func1 = RNGkutiSys(F,x0,y0,z0,n,step)

    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivodSys(n, func)
    xs,ys,zs = kk[0],kk[1],kk[2]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivodSys(n, func1)
    xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

    fig = plt.figure(figsize = (7,7))
    plt.title(f'exp(-(y+z))+2*x\nx**2+y\n y({x0}) = {y0}, z({x0}) = {z0}')
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
    ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
    ax.legend()
    fig.show()
    input('Введите любой символ чтобы продолжить >>> ')

    #Вариант 34
    print(' Вариант 34 '.center(45,'-'))
    F = lambdify([x,y], eval('-y/x-y**2*ln(x)'))
    x0 = 1
    y0 = 2
    step34 = (2 - 1)/n
    func = AYEler(F,x0,y0,n,step34)
    func1 = RNGkuti(F,x0,y0,n,step34)
    print(' Решение методом Эйлера-Коши '.center(45,'-'))
    kk = vivod(n, func)
    xs,ys = kk[0],kk[1]
    print(' Решение методом Рунге Кутты '.center(45,'-'))
    kk1 = vivod(n, func1)
    xs1,ys1 = kk1[0],kk1[1]
    plt.figure(figsize = (7,7))
    plt.title(f'exp(-x)*(y**2+1.04)\ny({x0}) = {y0}')
    plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
    plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
    plt.legend()
    plt.show()
    input('Введите любой символ чтобы продолжить >>> ')

#--- ОСНОВНОЙ КОД РЕШЕНИЯ ОДУ ---#
def ODU():
    """[Запуск программы по работе с ОДУ]

    """
    while (True):
        print(''.center(45,'-'))
        inpt = input('\nВыберите метод ввода :\n1)Одно уравнение\n2)Система уравнений\n3)Тестовые варианты\n\nВведите любой другой символ для вывхода из программы\n')
        if inpt == '1': 
            F = vvod()
            k_dec = decor()
            x0 = k_dec[0] 
            y0 = k_dec[1]
            a = k_dec[2]
            b = k_dec[3]
            n = k_dec[4]
            try:
                step = (b - a)/n
            except ZeroDivisionError:
                print('\nРешение невозможно. Кол-во шагов = 0')
                return ODU()
            try:
                func = AYEler(F,x0,y0,n,step)
                func1 = RNGkuti(F,x0,y0,n,step)
            except ZeroDivisionError:
                print('ОШИБКА!!! обнаружено деление на 0')
                return ODU()

            print(' Решение методом Эйлера-Коши '.center(45,'-'))
            kk = vivod(n, func)
            xs,ys = kk[0],kk[1]
            
            print("Macсив [x,y,y']")
            kk = y_sh(func,ys,step)
            s = 0                   #Cумма модулей
            for i in range(len(kk)):
                s += abs(kk[i][1])
            kk = trns(kk)
        
            print(' Решение методом Рунге Кутты '.center(45,'-'))
            kk1 = vivod(n, func1)
            xs1,ys1 = kk1[0],kk1[1]

            print("Macсив [x,y,y']")
            kk1 = y_sh(func1,ys1,step)
            s1 = 0                   #Cумма модулей
            for i in range(len(kk1)):
                s1 += abs(kk1[i][1])
            kk1 = trns(kk1)
            try:
                plt.figure(figsize = (8,8))
                plt.subplot(2,1,1)
                plt.xlabel('Ось X')
                plt.ylabel('Ось Y')
                plt.title('Решение ОДУ')
                plt.plot(xs, ys, label = 'Эйлер-Коши', c = 'r')
                plt.plot(xs1, ys1, label ='Рунге Кутты', c = 'b')
                plt.legend()

                plt.subplot(2,1,2)
                plt.xlabel('Значение X')
                plt.ylabel('Значение разности')
                plt.title("Разность между y' и функцией ")
                plt.plot(kk[0], kk[1], label = 'Эйлер-Коши', c = 'r')
                plt.plot(kk[0], kk[1], label = f'Сумма модулей = {s}', c = 'r')
                plt.plot(kk1[0], kk1[1], label = 'Рунге Кутты', c = 'b')
                plt.plot(kk1[0], kk1[1], label = f'Сумма модулей = {s1}', c = 'b')
                plt.legend()
                plt.show()
            except TypeError:
                print('Программа затрудняется построить график')
                return ODU()

        elif inpt == '2':
            F = vvod1()
            x0 = float(input('Введите x0 >>> '))
            y0 = float(input('Введите y0 >>> '))
            z0 = float(input('Введите z0 >>> '))
            a = int(input('Введите начальную точку интервала >>> '))
            b = int(input('Введите конечную точку интервала >>> '))
            n = int(input('Введите кол-во точек >>> '))
            step = (b - a)/n
            
            try:
                func = AYElerSys(F,x0,y0,z0,n,step)
                func1 = RNGkutiSys(F,x0,y0,z0,n,step)
            except ZeroDivisionError:
                print('ОШИБКА!!! обнаружено деление на 0')
                return ODU()

            print(' Решение методом Эйлера-Коши '.center(45,'-'))
            kk = vivodSys(n, func)
            xs,ys,zs = kk[0],kk[1],kk[2]

            print("Macсив [x,y,z,y',z']")
            kk = y_shSys(func,ys,zs,step)
            s = 0
            z = 0                  #Cумма модулей
            for i in range(len(kk)):
                s += abs(kk[i][1])
                z += abs(kk[i][2])
            kk = trns(kk)

            print(' Решение методом Рунге Кутты '.center(45,'-'))
            kk1 = vivodSys(n, func1)
            xs1,ys1,zs1 = kk1[0],kk1[1],kk1[2]

            print("Macсив [x,y,z,y',z']")
            kk1 = y_shSys(func1,ys1,zs1,step)
            s1 = 0
            z1 = 0                  #Cумма модулей
            for i in range(len(kk1)):
                s1 += abs(kk1[i][1])
                z1 += abs(kk1[i][2])
            kk1 = trns(kk1)

            fig = plt.figure(figsize = (7,7))
            ax = fig.add_subplot(projection='3d')
            #ax.title('Решение ОДУ')
            ax.plot(xs, ys, zs, label = 'Эйлер-Коши', c = 'r')
            ax.plot(xs1, ys1, zs1, label ='Рунге Кутты', c = 'b')
            ax.legend()
            '''
            bx = fig.add_subplot(projection='3d')
            #ax.title("Разность между y', z' и функцией ")
            bx.plot(kk[0], kk[1], kk[2], label = 'Эйлер-Коши', c = 'r')
            bx.plot(kk[0], kk[1], kk[2], label =f"Сумма модулей y' = {s}\nСумма модулей z' = {z}", c = 'r')
            bx.plot(kk1[0], kk1[1], kk1[2], label ='Рунге Кутты', c = 'b')
            bx.plot(kk1[0], kk1[1], kk1[2], label = f"Сумма модулей y' = {s1}\nСумма модулей z' = {z1}", c = 'b')
            bx.legend()
            '''
            fig.show()
        
        elif inpt == '3':
            tst()
        else:
            break


#-------------------- 7-8 --------------------#
# Функции создания матриц 
def mtx_Way(a):
    """[Генерация матрицы весовых коэф-в]
    -Используется для решения задач на оптимизацию
    -Диапазон генерации 1-50
    Args:
        a ([int]): [Ранг матрицы]

    Returns:
        [list]: [Матрица весовых коэф-в]
    """
    m1 = []
    for i in range(a):
        m2 = []
        for j in range(a):
            m2.append(randint(1,50))
        m1.append(m2)
        m1[i][i] = 0
    
    for i in range(len(m1)):
        for j in range(len(m1)):
            m1[j][i] = m1[i][j]
    out(m1)
    return m1
def mtxcreat():
    """[Создание матрицы весовых коэф-в]
    -Пользователь вводит количество городов(ранг матрицы)
    -Пользователь вводит значение поэлементно
    Returns:
        [list]: [Матрицы весовых коэф-в]
    """
    def cock():  #Создание матрицы 1
        try:
            num[i][j] = int(input())
            return num[i][j]
        except ValueError:
            print('Ошибка ввода!!!\nВведите элемент заново\n')
            return cock()
        
    try:
        x = int(input('\nВведите Кол-во городов >>>  '))
        num = [[0 for n in range(x)] for nn in range(x)]
        print('Введите значения матрицы поэлементно')
        for i in range(x):
            for j in range(x):
                if i!=j:
                    print(f'Введите длину пути из {i+1} в {j+1}')
                    cock()
        print('Вы ввели матрицу:')
        out(num) 
        return num 
    except ValueError:
        print('\n>>> Ошибка!!!\nВведите заново кол-во сток/столбцов\n')
        return(mtxcreat())

# Генераторы случайностей
def rndWay(a): #Случайный путь
    """[Случайный путь]

    Args:
        a ([list]): [Матрица весовых коэф-в]

    Returns:
        [list]: [Случайный путь]
    """
    b = [i+1 for i in range(a)]
    shuffle(b)
    b.append(b[0])
    return b
def genCh(way): #Случайные числа для замены
    b = [i+1 for i in range(len(way))]
    mm0 = b[1:-1]
    
    mm = sample(mm0,2)
    way[(mm[0])-1],way[(mm[1])-1] = way[(mm[1])-1],way[(mm[0])-1]
    return way
def summ_Way(m1,way): # Длина пути
    """[Длина пути]

    Args:
        m1 ([list]): [Матрица весовых коэф-в]
        way ([list]): [Путь]

    Returns:
        [float]: [Длина пути]
    """
    S = 0
    for i in range(len(way)-1):
        S += m1[(way[i])-1][(way[i+1])-1]
    return S

####### Муравьи #######
def GeneratorRex(n,Countries): # Авто генерация путей
        Wow = []
        Wow1 = []
        for i in range(1, n + 1):
            for x in range(1, n + 1):
                Countries[int(str(i) + str(x))] = int(randint(10, 20))
                if x == i:
                    Countries[int(str(i) + str(x))] = "0"
                q = 0
                if q == 1:
                    Countries[int(str(i) + str(x))] = "0"
                Wow.append(Countries[int(str(i) + str(x))])
            Wow1.append(Wow)
            Wow = []
        print('\nМатрица Путей')
        out(Wow1)
def MatrizaPher(n,Countries,MAtrizaPheromonov):
    Wow = []
    Wow1 = []
    for i in range(1, n + 1):
        for x in range(1, n + 1):
            MAtrizaPheromonov[int(str(i) + str(x))] = 3
            if x == i:
                MAtrizaPheromonov[int(str(i) + str(x))] = "0"
            q = randint(1, 5)
            if Countries[int(str(i) + str(x))] == "0":
                MAtrizaPheromonov[int(str(i) + str(x))] = 0
            Wow.append(MAtrizaPheromonov[int(str(i) + str(x))])
        Wow1.append(Wow)
        Wow = []
    print('\nМатрица феромонов')
    out(Wow1)
def GeneratorRex2(n,Countries):
    Wow = []
    Wow1 = []

    for i in range(1, n + 1):
        for x in range(1, n + 1):
            print(i, x, "введите:")
            Countries[int(str(i) + str(x))] = input()
            if x == i:
                Countries[int(str(i) + str(x))] = "a"
            Wow.append(Countries[int(str(i) + str(x))])
        Wow1.append(Wow)
        Wow = []
    for i in Wow1:
        print(i)

#--- ОСНОВНОЙ КОД РЕШЕНИЯ Оптимизации ---#
def Optimize():
    """[Запуск программы по Оптимизации]

    """
    while (True):
        cmd = input('\nВыберите метод Решения:\n1)Алгоритм имитации отжига\n2)Алгоритм муравьиной колонии\n\nВведите любой другой символ для вывхода из программы\n')
        if cmd == '1':
            while (True):
                print(''.center(45,'-'))
                inpt = input('\nВыберите метод ввода матрицы весовых коэф-в:\n1)Ввод вручную\n2)Случайная генерация\n3)Импорт CSV файла\n\nВведите любой другой символ чтобы вернуться назад\n')
                if inpt == '1':
                    mtx1 = mtxcreat()
                    if len(mtx1) == 0:
                        print('Городов нет :(\n Конец')
                        return Optimize()
                    way = rndWay(len(mtx1))
                
                    alph = 0.9 # Коэф. понижения температуры за итерацию
                    T = 100 #начальная температура
                    count = 0
                    countL = []
                    L = summ_Way(mtx1,way)
                    countL.append(L)
                    while T > 0.001:
                        count += 1
                        way = genCh(way)
                        ww = copy.deepcopy(way)
                        L1 = summ_Way(mtx1,way)

                        dL = L1 - L
                        P_form = 100*exp(-dL/T)

                        if dL < 0:
                            L = L1
                            T = T*alph
                            w1 = ww
                        elif P_form > randint(1,100):
                            L = L1
                            T = T*alph
                            w1 = ww
                        else:
                            T = T*alph
                        countL.append(L)
                    print(f'Кратчайший путь = {w1}, Длина = {L}')

                    plt.figure(figsize = (7,7))
                    plt.title('Решение методом отжига')
                    plt.plot(range(count+1), countL, label = 'Длина пути',c = 'r')
                    plt.xlabel('Кол-во Итераций')
                    plt.ylabel('Длина пути')
                    plt.legend()
                    plt.show()        
                elif inpt == '2':
                    a = int(input("Введите кол-во городов "))
                    if a == 0:
                        print('Городов нет :(\n Конец')
                        return Optimize()
                    mtx1 = mtx_Way(a)
                    way = rndWay(a)
                    
                    alph = 0.9 # Коэф. понижения температуры за итерацию
                    T = 100 #начальная температура
                    count = 0
                    countL = []
                    L = summ_Way(mtx1,way)
                    countL.append(L)
                    while T > 0.001:
                        count += 1
                        way = genCh(way)
                        ww = copy.deepcopy(way)
                        L1 = summ_Way(mtx1,way)

                        dL = L1 - L
                        P_form = 100*exp(-dL/T) #Формула вероятности

                        if dL < 0:
                            L = L1
                            T = T*alph
                            w1 = ww
                        elif P_form > randint(1,100):
                            L = L1
                            T = T*alph
                            w1 = ww
                        else:
                            T = T*alph
                        countL.append(L)
                    print(f'Кратчайший путь = {w1}, Длина = {L}')

                    plt.figure(figsize = (7,7))
                    plt.title('Решение методом отжига')
                    plt.plot(range(count+1), countL, label = 'Длина пути',c = 'r')
                    plt.xlabel('Кол-во Итераций')
                    plt.ylabel('Длина пути')
                    plt.legend()
                    plt.show()       
                elif inpt == '3':
                    mtx1 = CSV()
                    if len(mtx1) == 0:
                        print('Городов нет :(\n Конец')
                        return Optimize()
                    way = rndWay(len(mtx1))

                    alph = 0.9 # Коэф. понижения температуры за итерацию
                    T = 100 #начальная температура
                    count = 0
                    countL = []
                    L = summ_Way(mtx1,way)
                    countL.append(L)
                    while T > 0.001:
                        count += 1
                        way = genCh(way)
                        ww = copy.deepcopy(way)
                        L1 = summ_Way(mtx1,way)

                        dL = L1 - L
                        P_form = 100*exp(-dL/T)

                        if dL < 0:
                            L = L1
                            T = T*alph
                            w1 = ww
                        elif P_form > randint(1,100):
                            L = L1
                            T = T*alph
                            w1 = ww
                        else:
                            T = T*alph
                        countL.append(L)
                    print(f'Кратчайший путь = {w1}, Длина = {L}')

                    plt.figure(figsize = (7,7))
                    plt.title('Решение методом отжига')
                    plt.plot(range(count+1), countL, label = 'Длина пути',c = 'r')
                    plt.xlabel('Кол-во Итераций')
                    plt.ylabel('Длина пути')
                    plt.legend()
                    plt.show()
                else:
                    break
        elif cmd == '2':
            print('ДОСТУПНА ТОЛЬКО СЛУЧАЙНАЯ ГЕНЕРАЦИЯ МАТРИЦЫ\nДругие способы ввода будут доступны в следующих обновлениях ')
            n = int(input('\nВведите Кол-во городов >>>  '))

            Q = 300  # константа 
            a = 4  # константа 
            b = 2  # константа 
            Mnojest = {1}
            Inters = []
            Inters.append(1)
            Countries = {}  # Множество городов
            MAtrizaPheromonov = {}  # матрица феромончиков
            for i in range(2, n + 1):
                Mnojest.add(i)
            GeneratorRex(n,Countries)
            MatrizaPher(n,Countries,MAtrizaPheromonov)
            class DarkArmy():
                Piii = 1
                xfutlocation = 0
                yfuturelocation = 0
                Saving = copy.deepcopy(Mnojest)
                FutureCountries = []
                FutureCu = 0
                FutureCountriesP = []
                PastCountries = []
                Q = 300
                a = 4
                b = 2
                Y = 4
                Ready = 0
                countoffailures = 2

                def __init__(self, num):
                    self.Piii = num

                    self.Mnojest = {num}
                    self.PastCountries = [num]
                def choosetown(self):
                    if self.Ready == 0:
                        self.FutureCountries = []
                        self.FutureCountriesP = []
                        for i in range(1, n + 1):
                            if type(Countries[int(str(self.Piii) + str(i))]) is int and i not in self.PastCountries:
                                self.FutureCountries.append(int(i))
                        if not self.FutureCountries and self.Saving != self.Mnojest:
                            self.FutureCu = self.PastCountries[-(self.countoffailures)]

                            self.PastCountries.append(self.FutureCu)
                            self.Mnojest.add(self.FutureCu)
                            self.countoffailures += 1
                        else:
                            for i in self.FutureCountries:
                                self.FutureCountriesP.append(((Q / Countries[int(str(self.Piii) + str(i))]) ** a) * (
                                            (MAtrizaPheromonov[int(str(self.Piii) + str(i))]) ** b))
                            self.FutureCu = choices(self.FutureCountries, weights=self.FutureCountriesP, k=1)

                            self.PastCountries.append(self.FutureCu[0])
                            self.countoffailures = 2
                            self.Mnojest.add(self.FutureCu[0])
                        # print(self.PastCountries)

                def NextStep(self):
                    if self.Ready == 0:
                        self.xlocation = self.xfutlocation
                        self.ylocation = self.yfuturelocation

                def check(self):
                    if Mnojest == self.Mnojest:
                        self.Ready = 1

                    return self.Ready
            Nya1 = []
            Nya2 = []
            Lelwd3 = 0
            Counter = 1
            hmph = []
            MinID = 0
            for i in range(80):

                Anty = []
                for i in range(n + 1):
                    Anty.append(1)

                for i in range(1, n + 1):
                    Anty[i] = DarkArmy(num=i)
                while True:
                    for i in range(1, n + 1):
                        Anty[i].choosetown()
                    for i in range(1, n + 1):
                        Anty[i].NextStep()
                    AAA = 0
                    for i in range(1, n + 1):
                        AAA += Anty[i].check()

                    if AAA == n:
                        break
                    AAA = 0
                for i in range(1, n + 1):
                    for YYY in range(1, n):

                        Qar = copy.copy(Anty[i].PastCountries)

                        Qar.insert(0, 1)

                        if type(MAtrizaPheromonov[int(str(Qar[YYY]) + str(Qar[YYY + 1]))]) is int or float:
                            MAtrizaPheromonov[int(str(Qar[YYY]) + str(Qar[YYY + 1]))] = MAtrizaPheromonov[int(str(Qar[YYY]) + str(
                                Qar[YYY + 1]))] * 0.64 + Q / Countries[int(str(Qar[YYY]) + str(Qar[YYY + 1]))]

                Nya1 = copy.copy(Anty[1].PastCountries)
                hmph.append(Nya1)
                Lelwd3 = 0
                for i in range(len(Nya1) - 1):
                    Lelwd3 += Countries[int(str(Nya1[i]) + str(Nya1[i + 1]))]
                Nya2.append(Lelwd3)

                Counter += 1
            Checking = Nya2[0]
            for i in range(len(Nya2)):
                if Checking > Nya2[i]:
                    Checking = Nya2[i]
                    MinID = i
            print(f'Кратчайший путь = {hmph[MinID]}, Длина = {Checking}')
            print('\nРешение полученно при помощи ОАО"Александр Шелягин" :)')
            plt.figure(figsize = (5,5))
            plt.title('Алгоритм муравьиной колонии')
            plt.plot(range(Counter - 1), Nya2, label = 'Длина пути',c = 'r')
            plt.legend()
            plt.show()          
        else:
            break
