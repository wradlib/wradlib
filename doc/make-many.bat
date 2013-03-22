set pythonpath=D:\pfaff\projekte\software\wradlib-outgoing
set SPHINXOPTS=-D pngmath_latex=D:\pfaff\tools\miktex-portable\miktex\bin\latex.exe -D pngmath_dvipng=D:\pfaff\tools\miktex-portable\miktex\bin\dvipng.exe
call make.bat clean
deltree source/generated
call make.bat htmlhelp > make-htmlhelp.log 2>&1
call make.bat html > make-html.log 2>&1
pause