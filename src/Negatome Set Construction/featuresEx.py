from splinter import Browser
import time
import csv
import os
p = []
pUnip = []
for i in range(1,6446): #6445):
    with open("Negatomeset\\"+str(i)+".fasta") as file:
        proteins = file.readlines()
        protA = ""
        protB = ""
        pUnipA = ""
        pUnipB = ""
        now = -1
        for l in proteins:
            if l[0] == '>':
                if now == -1:
                    now = 1
                    pUnipA = l.split("|")[1]
                    #print("Uniprot ID A: " + pUnipA)
                else:
                    now = 2
                    pUnipB = l.split("|")[1]
                    #print("Uniprot ID B: " + pUnipB)
                continue
            else:
                if now == 1:
                    protA = protA + l
                else:
                    protB = protB + l
        if pUnipA not in pUnip:
            pUnip.append(pUnipA)
            p.append(protA)
        if pUnipB not in pUnip:
            pUnip.append(pUnipB)
            p.append(protB)
print("number of proteins: " + str(len(p)))
all_features = []
with Browser('chrome') as browser:
    # Visit URL
    url = "http://bidd2.nus.edu.sg/cgi-bin/prof2015/protein/profnew.cgi"
    for i in range(601, len(p)):
        try:
            if i % 5 == 0:
                print(i)
            browser.visit(url)

            # fill the query form with our search term
            browser.fill('sequence', p[i])

            # find the search button on the page and click it
            button = browser.find_by_name('submit')
            button.click()

            radiobutton = browser.find_by_value('CSV_D')
            radiobutton.click()

            button = browser.find_by_name('Submit1')
            button.click()

            link = browser.find_link_by_partial_href(".out")
            href = link['href']
            filename = href.split("file=",1)[1]
            #print(filename)

            link.click()
            while True:# wait for download
                time.sleep(0.5)
                if os.path.isfile(filename):
                    break
            with open(filename) as file:
                lines = file.readlines()
                features = lines[1].split(",")
                features = features[1:421] + features[691:1435]
                #print(len(features))
                all_features.append([pUnip[i]]+ features)
            os.remove(filename)
        except:
            print("An error occured with protein with UniProt: " + p[i])
            #continue
        if i % 100 == 0 or i == len(p)-1: # save every 100 samples for backup
            with open("path\to\downloads\\features"+str(i)+".csv", "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(all_features)
