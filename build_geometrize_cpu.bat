@echo off
setlocal

set ROOT=%~dp0
set OUT_DIR=%ROOT%build\geometrize_cpu
set OBJ_DIR=%OUT_DIR%\obj
set GEOM_SRC=%ROOT%third-party\geometrize\lib\geometrize\geometrize
set VC_VARS=

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set VC_VARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if not defined VC_VARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set VC_VARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if not defined VC_VARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set VC_VARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat
if not defined VC_VARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set VC_VARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat

if not defined VC_VARS (
  echo Could not find vcvars64.bat
  exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"
call "%VC_VARS%" >nul

cl /nologo /std:c++17 /EHsc /O2 /DNDEBUG ^
  /I"%GEOM_SRC%" ^
  /Fo"%OBJ_DIR%\\" ^
  "%ROOT%tools\geometrize_cpu_runner.cpp" ^
  "%GEOM_SRC%\geometrize\bitmap\bitmap.cpp" ^
  "%GEOM_SRC%\geometrize\bitmap\rgba.cpp" ^
  "%GEOM_SRC%\geometrize\exporter\bitmapdataexporter.cpp" ^
  "%GEOM_SRC%\geometrize\exporter\bitmapexporter.cpp" ^
  "%GEOM_SRC%\geometrize\exporter\shapearrayexporter.cpp" ^
  "%GEOM_SRC%\geometrize\exporter\shapejsonexporter.cpp" ^
  "%GEOM_SRC%\geometrize\exporter\shapeserializer.cpp" ^
  "%GEOM_SRC%\geometrize\exporter\svgexporter.cpp" ^
  "%GEOM_SRC%\geometrize\rasterizer\rasterizer.cpp" ^
  "%GEOM_SRC%\geometrize\rasterizer\scanline.cpp" ^
  "%GEOM_SRC%\geometrize\runner\imagerunner.cpp" ^
  "%GEOM_SRC%\geometrize\shape\circle.cpp" ^
  "%GEOM_SRC%\geometrize\shape\ellipse.cpp" ^
  "%GEOM_SRC%\geometrize\shape\line.cpp" ^
  "%GEOM_SRC%\geometrize\shape\polyline.cpp" ^
  "%GEOM_SRC%\geometrize\shape\quadraticbezier.cpp" ^
  "%GEOM_SRC%\geometrize\shape\rectangle.cpp" ^
  "%GEOM_SRC%\geometrize\shape\rotatedellipse.cpp" ^
  "%GEOM_SRC%\geometrize\shape\rotatedrectangle.cpp" ^
  "%GEOM_SRC%\geometrize\shape\shapefactory.cpp" ^
  "%GEOM_SRC%\geometrize\shape\shapemutator.cpp" ^
  "%GEOM_SRC%\geometrize\shape\shapetypes.cpp" ^
  "%GEOM_SRC%\geometrize\shape\triangle.cpp" ^
  "%GEOM_SRC%\geometrize\commonutil.cpp" ^
  "%GEOM_SRC%\geometrize\core.cpp" ^
  "%GEOM_SRC%\geometrize\model.cpp" ^
  "%GEOM_SRC%\geometrize\state.cpp" ^
  /Fe"%OUT_DIR%\geometrize_cpu_runner.exe"

if errorlevel 1 exit /b 1
echo Built "%OUT_DIR%\geometrize_cpu_runner.exe"
