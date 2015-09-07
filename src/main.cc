
//
#include "app.h"
#include "hellomodule.h"
#include "foobarmodule.h"

// STL
#include <iostream>

//
#include <unistd.h>

using namespace std;

int main( int argc, char** argv )
{
	App a;
	a.addModule( "Hello", new HelloModule() );
	a.addModule( "FooBar", new FoobarModule() );

	a.modules("Hello")->addRecipient("FooBar");

	a.start();

	sleep( 10 );

	a.stop();

	sleep( 1 );

	return 0;
}
