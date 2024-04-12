from flask import Flask,render_template,request
import pickle
import warnings
warnings.filterwarnings("ignore")
applicaton=Flask(__name__)
app=applicaton

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    else:
        school=request.form.get('school')
        sex=request.form.get('sex')
        age=int(request.form.get('age'))
        address=request.form.get('address')
        famsize=request.form.get('famsize')
        Pstatus=request.form.get('Pstatus')
        Medu=int(request.form.get('Medu'))
        Fedu=int(request.form.get('Fedu'))
        Mjob=request.form.get('Mjob')
        Fjob=request.form.get('Fjob')
        guardian=request.form.get('guardian')
        traveltime=int(request.form.get('traveltime'))
        studytime=int(request.form.get('studytime'))
        failures=int(request.form.get('failures'))
        nursery=request.form.get('nursery')
        internet=request.form.get('internet')
        freetime=int(request.form.get('freetime'))
        goout=int(request.form.get('goout'))
        health=int(request.form.get('health'))
        absence=int(request.form.get('absence'))
        G1=int(request.form.get('G1'))
        G2=int(request.form.get('G2'))
        def yesNo_to_num(yesno):
            return 0 if yesno=='no' else 1
        def cat_to_num(school,sex,address,famsize,Pstatus,Mjob,Fjob,guardian,nursery,internet):
            school=0 if school=='GP' else 1
            sex=1 if sex=='M' else 0
            address=0 if address=='R' else 1
            famsize=0 if famsize=='GT3' else 1
            Pstatus=0 if Pstatus=='A' else 1
            if Mjob=='at_home':
                Mjob=0
            elif Mjob=='services':
                Mjob=3
            elif Mjob=='teacher':
                Mjob=4
            elif Mjob=='health':
                Mjob=1
            else:
                Mjob=2
            # for Fjob
            if Fjob=='at_home':
                Fjob=0
            elif Fjob=='services':
                Fjob=3
            elif Fjob=='teacher':
                Fjob=4
            elif Fjob=='health':
                Fjob=1
            else:
                Fjob=2
            # reason
            # guardian
            if guardian=='mother':
                guardian=1
            else:
                guardian=0
            # schoolsup=yesNo_to_num(schoolsup)
            # famsup=yesNo_to_num(famsup)
            # paid=yesNo_to_num(paid)
            # activities=yesNo_to_num(activities)
            nursery=yesNo_to_num(nursery)
            # higher=yesNo_to_num(higher)
            internet=yesNo_to_num(internet)
            # romantic=yesNo_to_num(romantic)
            return (school,sex,address,famsize,Pstatus,Mjob,Fjob,guardian,
                       nursery,internet)

        values=cat_to_num(school,sex,address,famsize,Pstatus,Mjob,Fjob,guardian,
                    nursery,internet)
        (school,sex,address,famsize,Pstatus,Mjob,Fjob,guardian,
                    nursery,internet)=values
        
        def predict_machine_failure(model,parameters):
                predicted=model.predict([parameters])
                return predicted
        parameters=[school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,guardian,traveltime,
         studytime,failures,nursery,internet,freetime,goout,health,absence,G1,G2]
        # print(parameters)
        loaded_model = pickle.load(open('reg_model.pkl', 'rb'))
        predicted_result=predict_machine_failure(loaded_model,parameters)[0]
        predicted_result=round(predicted_result,2)
        tip=''
        if traveltime>3:
            tip+='Decrease TravelTime\n'
            tip+=' and '
        if studytime<=2:
            tip+='Increase StudyTime\n'
            tip+=' and '
        if failures>0:
            tip+='Try to Decrease Failures !!\n'
            tip+=' and '
        if freetime>=3:
            tip+='Your FreeTime is High, try to invest it in Study\n'
            tip+=' and '
        if goout>=3:
            tip+='Your Go out Time is High,Lower it and Increase StudyTime\n'
            tip+=' and '
        if absence>=8:
            tip+='Your Absences are high,Come regularly to college!\n'
        # print(tip)
        return render_template('index.html',result=predicted_result,tip=tip)
        
if __name__=='__main__':
    app.run(host='0.0.0.0')

