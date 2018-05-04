========================================================================
    DYNAMIC LINK LIBRARY : FastTreeNative Project Overview
========================================================================

This file contains a summary of what you will find in each of the files that
make up your FastTreeNative application.

FastTreeNative.cpp
    This is the main DLL source file.

	When created, this DLL does not export any symbols. As a result, it
	will not produce a .lib file when it is built. If you wish this project
	to be a project dependency of some other project, you will either need to
	add code to export some symbols from the DLL so that an export library
	will be produced, or you can set the Ignore Input Library property to Yes
	on the General propert page of the Linker folder in the project's Property
	Pages dialog box.

/////////////////////////////////////////////////////////////////////////////
Other standard files:

StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named FastTreeNative.pch and a precompiled types file named StdAfx.obj.

/////////////////////////////////////////////////////////////////////////////
