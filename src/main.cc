
//
#include "app.h"
#include "hellomodule.h"
#include "foobarmodule.h"
#include "scriptedmodule.h"

// STL
#include <iostream>

//
#include <unistd.h>

using namespace std;

int main( int argc, char** argv )
{


      
	App a;
	
    a.addModule( "Test", new ScriptedModule("my popo source code") );
	a.addModule( "Hello", new HelloModule() );
	a.addModule( "FooBar", new FoobarModule() );
    a.addModule( "FooBar1", new FoobarModule() );
    a.addModule( "FooBar2", new FoobarModule() );
    a.addModule( "FooBar3", new FoobarModule() );
	a.addModule( "FooBar4", new FoobarModule() );
	a.addModule( "FooBar5", new FoobarModule() );
	a.addModule( "FooBar6", new FoobarModule() );
	a.addModule( "FooBar7", new FoobarModule() );
	a.addModule( "FooBar8", new FoobarModule() );
	a.addModule( "FooBar9", new FoobarModule() );
	
    a.modules("Hello")->addRecipient("Test");
    //a.modules("Hello")->addRecipient("FooBar");
    a.modules("Test")->addRecipient("FooBar");
    a.modules("Test")->addRecipient("FooBar2");
    a.modules("Test")->addRecipient("FooBar3");
    a.modules("Test")->addRecipient("FooBar4");
    a.modules("Test")->addRecipient("FooBar5");
    a.modules("Test")->addRecipient("FooBar6");
    a.modules("Test")->addRecipient("FooBar7");
    a.modules("Test")->addRecipient("FooBar8");
    a.modules("Test")->addRecipient("FooBar9");
	/*
	a.modules("Hello")->addRecipient("Test1");
	a.modules("Hello")->addRecipient("Test2");
	a.modules("Hello")->addRecipient("Test3");
	a.modules("Hello")->addRecipient("Test4");
	a.modules("Hello")->addRecipient("Test5");
	a.modules("Hello")->addRecipient("Test6");
	*/
	
	// a.modules("Hello")->addRecipient("FooBar2");
	// a.modules("Hello")->addRecipient("FooBar3");
	/*
	a.modules("Hello")->addRecipient("FooBar4");
	a.modules("Hello")->addRecipient("FooBar5");
	a.modules("Hello")->addRecipient("FooBar6");
	a.modules("Hello")->addRecipient("FooBar7");
	a.modules("Hello")->addRecipient("FooBar8");
	a.modules("Hello")->addRecipient("FooBar9");
	*/
	
	while(1)
	{
		a.start();
		sleep( 8 );
		a.stop();
		sleep( 4 );
	}


	return 0;
}
