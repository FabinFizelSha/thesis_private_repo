# Thesis template for student thesis at IAS, University of Stuttgart

This template is aimed to provide a LaTeX writing experience with low hurdles.

The template is based on the thesis template from the ISW, University of Stuttgart at overleaf <https://www.overleaf.com/latex/templates/isw-student-thesis/xjwkfntnwrwc>.

## How to use it:

1. Set up the project
    1. Rename the main file ```Name_Thema.tex``` to something useful.
    2. Edit the ```settings.tex```
        * select your language setup:
            * ```\usepackage[ngerman, english]{babel}``` if you write mainly in English
            * ```\usepackage[englisch, ngerman]{babel}``` if you write mainly in German
        * select your thesis type:
            * ```\usepackage[type=bachelor]{iasthesis}``` if you write a Bachelor Thesis
            * ```\usepackage[type=master]{iasthesis}``` if you write a Master Thesis
            * ```\usepackage[type=study]{iasthesis}``` if you write a "Forschungsarbeit"
    3. Set some variables in your main tex file ```Name_Thema.tex```
        * add your name in ```\author{}```
        * add your major in ```\major{}```
        * add the title of your thesis in ```\title{}```
        * add the thesis number in ```\thesisno{}```
        * add the submission date to ```\date{}```
        * add your examining professor to ```\professor{}```
        * add your supervisor to ```\supervisor{}```
        * if you have an external supervisor add his name to ```\extsupervisor{}``` and the company of your external supervisor to ```\extcompany{}```
        * if your thesis is confidental, set the ```\confidentialthesis{}``` variable

2. Write your thesis by adding your text to files in the ```chapters``` folder.

