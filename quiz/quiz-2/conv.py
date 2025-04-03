import json
import sys

if __name__ == "__main__":
    fname = sys.argv[1]
    fout = open("hhsjjs.html","w")
    with open(fname) as fhnd:
        fout.write("<!DOCTYPE html>\n")
        fout.write("<html>\n<head>\n</head>\n<body>\n")
        fout.write("<h1> QUIZ - 02 </h1>\n")
        fout.write("<div>\n")
        data = json.load(fhnd)        
        for d in data:
            fout.write("<details>\n")
            fout.write(f"<summary>{d['qno']}</summary>\n")
            fout.write(f"<p>{d['question']}</p>\n")
            if d["fig"]:
                fout.write(f"<img src=\"{d['fig']}\"></img>\n")
            fout.write("<ul>\n")
            #print(d)
            for i in d["mcq"]:
                fout.write(f"<li>{i}</li>")
            fout.write("</ul>\n")
            fout.write("</details>\n")
        fout.write("</div>\n")
        fout.write("</body>\n")
        fout.write("</html>\n")
    fout.close()