#pragma once

#include "module.h"

#include <sstream>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <random>

class HelloModule : public Module
{
	public:
		HelloModule()
		{
			srand( time(NULL) );
		}

		virtual ~HelloModule()
		{

		}

		void tick()
		{
			std::random_device rd;
                        std::mt19937 gen(rd());
                        std::exponential_distribution<double> dis(3.5);
			std::stringstream sstr;
			sstr << dis(gen);
                        Message msg(sstr.str(),"ppppp");
			emit( msg );
		}

	private:

	protected:


};
