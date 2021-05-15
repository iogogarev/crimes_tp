import pandas as pd
import numpy as np
import datetime
import calendar
import pickle
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
import sys
import os.path

def parse_date(s):
    res = {}
    try:
        dates = s.split()
        year = int((dates[0].split('-'))[0])
        res['year'] = year
        month = int((dates[0].split('-'))[1])
        res['month'] = month
        day = int((dates[0].split('-'))[2])
        res['day'] = day
        hour = int((dates[1].split(':'))[0])
        res['hour'] = hour
        minute = int((dates[1].split(':'))[1])
        res['minutes'] = minute

        if(minute == 30 or minute == 0):
            res['beautiful_endings'] = 1
        else:
            res['beautiful_endings'] = 0

        weekday = calendar.day_abbr[datetime.date(year, month, day).weekday()]
        res['weekday'] = weekday
        if(weekday == 'Sun' or weekday == 'Sat'):
            res['is_weekends'] = 1
        else:
            res['is_weekends'] = 0
        if(hour > 22 or (hour < 6)):
            res['nights'] = 1
            res['mornings'] = 0
            res['middle_days'] = 0
            res['afternoons'] = 0
        elif(hour >=6 and hour <= 10):
            res['nights'] = 0
            res['mornings'] = 1
            res['middle_days'] = 0
            res['afternoons'] = 0
        elif(hour > 10 and hour <=17):
            res['nights'] = 0
            res['mornings'] = 0
            res['middle_days'] = 1
            res['afternoons'] = 0
        else:
            res['nights'] = 0
            res['mornings'] = 0
            res['middle_days'] = 0
            res['afternoons'] = 1
    except:
        print('Wrong date format (correct : \'yyyy:mm:dd hh:ss\')')
        return None
    return res


def make_zones(X, Y, maxX = -122.365240723693, maxY = 37.819975492297004,
               minX = -122.513642064265, minY = 37.7078790224135):
    x_spread = maxX - minX
    y_spread = maxY - minY
    x_step = x_spread/15
    y_step = y_spread/15
    x_zone = int((X - minX)/x_step)
    y_zone = int((Y - minY)/y_step)
    if(x_zone == 15):
        x_zone = 14
    if(y_zone == 15):
        y_zone = 14
    return (15*x_zone+y_zone)

def parse_info(X, Y, date, district_clf, clf):
    try:
        X = float(X)
    except:
        print("X must be float (first argument)!")
        return
    try:
        Y = float(Y)
    except:
        print("Y must be float (second argument)!")
        return


    Y = float(Y)
    my_file = open("columns.txt")
    cols = my_file.read()
    my_file.close()
    columns = np.array(cols.split('\n'))
    arr = np.zeros(224)
    line = pd.DataFrame([arr], columns = columns)
    d = {}
    res = parse_date(date)
    if not res:
        return
    line['X'][0] = X
    line['Y'][0] = Y

    line['BeautifulEndings'][0] = res['beautiful_endings']
    line['IsWeekend'][0] = res['is_weekends']
    line['Nights'][0] = res['nights']
    line['Mornings'][0] = res['mornings']
    line['MiddleDays'][0] = res['middle_days']
    line['Afternoons'][0] = res['afternoons']
    line['Day'][0] = res['day']
    line['Months_' + str(res['month'])] = 1
    line['Hours'] = res['hour']


    if(res['weekday'] == 'Fri'):
        line['DayOfWeek_Friday'][0] = 1
    elif(res['weekday'] == 'Mon'):
        line['DayOfWeek_Monday'][0] = 1
    elif(res['weekday'] == 'Sat'):
        line['DayOfWeek_Saturday'][0] = 1
    elif(res['weekday'] == 'Sun'):
        line['DayOfWeek_Sunday'][0] = 1
    elif(res['weekday'] == 'Thu'):
        line['DayOfWeek_Thursday'][0] = 1
    elif(res['weekday'] == 'Tue'):
        line['DayOfWeek_Tuesday'][0] = 1
    elif(res['weekday'] == 'Wed'):
        line['DayOfWeek_Wednesday'][0] = 1

    district = district_clf.predict([[X, Y]])
    if(district == 'BAYVIEW'):
        line['PdDistrict_BAYVIEW'][0] = 1
    elif(district == 'CENTRAL'):
        line['PdDistrict_CENTRAL'][0] = 1
    elif(district == 'INGLESIDE'):
        line['PdDistrict_INGLESIDE'][0] = 1
    elif(district == 'MISSION'):
        line['PdDistrict_MISSION'][0] = 1
    elif(district == 'PARK'):
        line['PdDistrict_PARK'][0] = 1
    elif(district == 'RICHMOND'):
        line['PdDistrict_RICHMOND'][0] = 1
    elif(district == 'SOUTHERN'):
        line['PdDistrict_SOUTHERN'][0] = 1
    elif(district == 'TARAVAL'):
        line['PdDistrict_TARAVAL'][0] = 1
    elif(district == 'TENDERLOIN'):
        line['PdDistrict_TENDERLOIN'][0] = 1


    zone = make_zones(X, Y)
    if('Zones_' + str(zone) in columns):
        line['Zones_' + str(zone)] = 1

    maxX = -122.365240723693
    maxY = 37.819975492297004
    minX = -122.513642064265
    minY = 37.7078790224135

    line['XY1'][0] = (X - minX)**2 + (Y - minY)**2
    line['XY2'][0] = (maxX - X)**2 + (Y - minY)**2
    line['XY3'][0] = (X - minX)**2 + (maxY - Y)**2
    line['XY4'][0] = (maxX - X)**2 + (maxY - Y)**2
    line['XY45_2'][0] = Y * np.cos(np.pi / 4) - X * np.sin(np.pi / 4)
    line['XY30_1'][0] = X * np.cos(np.pi / 6) + Y * np.sin(np.pi / 6)
    line['XY30_2'][0] = Y * np.cos(np.pi / 6) - X * np.sin(np.pi / 6)
    line['XY60_1'] = X * np.cos(np.pi / 3) + Y * np.sin(np.pi / 3)

    line['XY60_2'] = Y * np.cos(np.pi / 3) - X * np.sin(np.pi / 3)

    X_median = -122.416452065595
    Y_median = 37.775420706711

    line['XY5'][0] = (X - X_median) ** 2 + (Y - Y_median) ** 2

    line['XY_rad'][0] = np.sqrt(np.power(Y, 2) + np.power(X, 2))
    d = {}
    predicted = clf.predict_proba(line)[0]
    for i in range(len(clf.classes_)):
        d[clf.classes_[i]] = predicted[i]
    sorted_dict = {}
    sorted_keys = sorted(d, key=d.get)

    for w in sorted_keys:
        sorted_dict[w] = d[w]
    s = ''
    keys = list(sorted_dict.keys())
    values = list(sorted_dict.values())
    for i in range(3):
        s += str(keys[len(keys) - 1 - i]) + ':' + str(values[len(keys) - 1 - i]) + '\n'

    return s

if not os.path.exists('model.joblib'):
    print('File model.joblib not found')

elif not os.path.exists('district_model.joblib'):
    print('File district_model.joblib not found')

elif not os.path.exists('columns.txt'):
    print('File columns.txt not found')

else:
    clf = load('model.joblib')
    district_clf = load('district_model.joblib')
    arguments = sys.argv[1:]
    f = open("info.txt", "w")
    answer = parse_info(arguments[0], arguments[1], arguments[2] + ' ' + arguments[3], district_clf, clf)
    if(answer):
        f.write(answer)
    f.close()
