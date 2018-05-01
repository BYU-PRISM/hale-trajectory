# -*- coding: utf-8 -*-

import glob
import datetime
import os

def updateStatus(name,hour,percent,iteration,failed,last_ten,TE,runTime,description):
    
    if(failed<10 and last_ten<5):
        color = 'panel-success'
    elif(last_ten<5):
        color = 'panel-default'
    elif(last_ten<10):
        color = 'panel-warning'
    else:
        color = 'panel-danger'
    
    template = """<?php
    echo "<div class='row'>";
    	echo "<div class='col-sm-12'>";
    		echo "<div class='panel {color}'>";
    		  echo "<div class='panel-heading'><b>{name} {description}</b></div>";
    		  echo "<div class='panel-body'>";
    			 echo "<div class='progress'>";
    			  echo "<div class='progress-bar' role='progressbar' aria-valuenow='{percent}' aria-valuemin='0' aria-valuemax='100' style='width:{percent}%'>";
    				echo "{hour} Hours Past Dawn";
    			  echo "</div>";
    			echo "</div>";
    			echo "<div class='col-sm-2 text-center'>";
    				echo "<b>Current Iteration:</b>";
    			echo "</div>";
    			echo "<div class='col-sm-1 text-center'>";
    				echo "{iteration}";
    			echo "</div>";
    			echo "<div class='col-sm-2 text-center'>";
    				echo "<b>Failed Solves</b>";
    			echo "</div>";
    			echo "<div class='col-sm-1 text-center'>";
    				echo "{failed}";
    			echo "</div>";
    			echo "<div class='col-sm-1 text-center'>";
    				echo "<b>Last 10</b>";
    			echo "</div>";
    			echo "<div class='col-sm-1 text-center'>";
    				echo "{last_ten}";
    			echo "</div>";
    			echo "<div class='col-sm-1 text-center'>";
    				echo "<b>TE</b>";
    			echo "</div>";
    			echo "<div class='col-sm-1 text-center'>";
    				echo "{energy}";
    			echo "</div>";
              echo "<div class='col-sm-1 text-center'>";
    				echo "<b>Run</b>";
    			echo "</div>";
    			echo "<div class='col-sm-1 text-center'>";
    				echo "{run} Hrs";
    			echo "</div>";
    		  echo "</div>";
    		echo "</div>";
    	echo "</div>";
    echo "</div>";
    ?>
    """
    
    context = {
     "name":name, 
     "hour":hour,
     "percent":percent,
     "iteration": iteration,
     "failed" : failed,
     "last_ten" : last_ten,
     "energy": TE,
     "run": runTime,
     "description": description,
     "color": color
     } 
    
    with  open('J:/groups/hale-models/www/apps/'+name+'.php','w') as myfile:
        myfile.write(template.format(**context))
        
    # Clean up old status files
    files = glob.glob('J:/groups/hale-models/www/apps/*.php')
    for file in files:
        mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(file))
        now = datetime.datetime.now()
        diff = now - mtime
#        print(abs(diff.total_seconds()/3600))
        if(abs(diff.total_seconds()/3600) > 24):
            os.remove(file)
        
if __name__ == "__main__":
    name = 'Application_1'
    hour = 1
    percent = 50
    iteration = 1283
    failed = 10
    last_ten = 10
    TE = 74
    runTime = 1
    description = 'Test'
    updateStatus(name,hour,percent,iteration,failed,last_ten,TE,runTime,description)