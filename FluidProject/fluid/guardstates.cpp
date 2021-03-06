#include "guardstates.h"
#include "guard.h"

extern Profiler GlobalProfiler;
extern CPUProfiler GlobalCPUProfiler;
GuardState* GuardStateToInit::statehandler = nullptr;
GuardState* GuardStateToStart::statehandler = nullptr;
GuardState* GuardStateToComplete::statehandler = nullptr;
GuardState* GuardStateToCheck::statehandler = nullptr;
GuardState* GuardStateToEndcheck::statehandler = nullptr;
GuardState* GuardStateToPause::statehandler = nullptr;
GuardState* GuardStateToTerminate::statehandler = nullptr;
GuardState* GuardStateToWait::statehandler = nullptr;
GuardState* GuardStateFactory::statehandler = nullptr;


std::string GuardStateToInit::statename = "Init";
std::string GuardStateToStart::statename = "Start";
std::string GuardStateToComplete::statename = "Complete";
std::string GuardStateToCheck::statename = "Check";
std::string GuardStateToEndcheck::statename = "Endcheck";
std::string GuardStateToPause::statename = "Pause";
std::string GuardStateToTerminate::statename = "Terminate";
std::string GuardStateToWait::statename = "Wait";
std::string GuardStateToResume::statename = "Resume";

//CPUTimer GuardStateToInit::cputimer;
//CPUTimer GuardStateToStart::cputimer;
//CPUTimer GuardStateToComplete::cputimer;
//CPUTimer GuardStateToCheck::cputimer;
//CPUTimer GuardStateToPause::cputimer;
//CPUTimer GuardStateToTerminate::cputimer;
//CPUTimer GuardStateToResume::cputimer;
//
//WallTimer GuardStateToInit::walltimer;
//WallTimer GuardStateToStart::walltimer;
//WallTimer GuardStateToComplete::walltimer;
//WallTimer GuardStateToCheck::walltimer;
//WallTimer GuardStateToPause::walltimer;
//WallTimer GuardStateToTerminate::walltimer;
//WallTimer GuardStateToResume::walltimer;

std::string get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	int total_time = (tv.tv_sec % 1000) * 1000000 + tv.tv_usec;
	return " "+std::to_string(total_time);
}

class GuardStateActions {
public:
	//void ActionAtCompleted(Guard* g);
	//void ActionAtTerminated(Guard* g);
	//void ActionAtChecking(Guard* g);
	//void ActionAtPausing(Guard* g);

	static void ActionAtCompleted(Guard* g) {
		GlobalCPUProfiler.addThread();
		////Handle the signal...
		//auto producer_set = g->producers->getSet();

		//for (auto producer : producer_set) {
		//	Signal sig_terminate("Send to producers");
		//	std::cout << g->name << " Sending " << sig_terminate.msg() << std::endl;
		//	putSignal(producer, sig_terminate);
		//}
		Signal sig_check("Check");
		//Guard * nextGuard = g->gs->nextGuardToStart(g);

		SyncLogger::print("COMPLETE>>>>>>>>>>", g->name, "Wall: ", (g->TaskTimerWT).stop(), "CPU:", (g->TaskTimerCT).stop());
		GuardScheduler::putGuard(g->gs, g);
		GuardScheduler::putGuard(g->gs, g, "Complete");
		//if (nextGuard) Guard::putSignal(nextGuard, sig_check);
		//if (nextGuard) SyncLogger::print(g->name, "send check to ", nextGuard->name);
		
		g->terminate = true;
		g->task->finish = true;
		for (Data *output_data : g->task->output) {
			output_data->set_tag(1);
		}
		auto producer_set = g->producers->getSet();
		Signal sig_terminate("Terminate");
		for (auto & producer : producer_set) {
			//			std::cout << g->name << " Sending " << sig_terminate.msg() << std::endl;
			//123///Guard::putSignal(producer, sig_terminate);
			producer->putSignal(producer, sig_terminate);

		}
		
		GlobalCPUProfiler.endThread();
		g->tag_log(g->name+" left State::Completed"+get_time());
	}
	static void ActionAtTerminated(Guard* g) {
		//Handle the signal...
		//auto producer_set = g->producers->getSet();

		//for (auto producer : producer_set) {
		//	Signal sig_terminate("Terminate");
		//	//			std::cout << g->name << " Sending " << sig_terminate.msg() << std::endl;
		//	Guard::putSignal(producer, sig_terminate);
		//}
		GlobalCPUProfiler.addThread();
		if(!g->terminate) {
			g->terminate = true;
			g->worker->terminate();
		}
		g->task->finish = true;
		for (Data *output_data : g->task->output) {
			output_data->set_tag(1);
		}
		SyncLogger::print("TERMINATE+++++++", g->name, "Wall: ", (g->TaskTimerWT).stop(), "CPU:", (g->TaskTimerCT).stop());
		//Guard * nextGuard = g->gs->nextGuardToStart(g);


		GuardScheduler::putGuard(g->gs, g);
		//if (nextGuard) Guard::putSignal(nextGuard, sig_check);
		//if (nextGuard) SyncLogger::print(g->name, "send check to ", nextGuard->name);
		//Guard::putSignal(g, "Completed");
		//Signal sig_check("Check");
		GlobalCPUProfiler.endThread();
		g->tag_log(g->name+" left State::Completed"+get_time());
	}
	static void ActionAtSpecTerminated(Guard* g) {
		//Handle the signal...
		//auto producer_set = g->producers->getSet();

		//for (auto producer : producer_set) {
		//	Signal sig_terminate("Terminate");
		//	//			std::cout << g->name << " Sending " << sig_terminate.msg() << std::endl;
		//	Guard::putSignal(producer, sig_terminate);
		//}
		GlobalCPUProfiler.addThread();
		//g->terminate = true;
		g->worker->terminate();
		/*g->task->finish = true;*/
		SyncLogger::print("ActionAtSpecTerminated+++++++", g->name, "Wall: ", (g->TaskTimerWT).stop(), "CPU:", (g->TaskTimerCT).stop());
		//Guard * nextGuard = g->gs->nextGuardToStart(g);


		GuardScheduler::putGuard(g->gs, g, "Terminate");
		//if (nextGuard) Guard::putSignal(nextGuard, sig_check);
		//if (nextGuard) SyncLogger::print(g->name, "send check to ", nextGuard->name);
		//Guard::putSignal(g, "Completed");
		//Signal sig_check("Check");
		GlobalCPUProfiler.endThread();
	}

	static void ActionAtCheckingAndTerminate(Guard* g) { //123//switch to ActionAtCheckingNoStarting
		GlobalCPUProfiler.addThread();
		//////std::cout << g->name << "static void ActionAtCheckingNoStarting" << std::endl;
		while (1) {
			bool check = true;
			//std::cout << g->name << " before checking valves" << std::endl;
			while (check) {
				check = false;
				for (auto& v : g->vs) {
					if (v->check() == false)
						check = true;
				}
			}
			//for (auto& v : g->vs) {
			//	std::cout << "RRRRRRRRRRRRRRRRR  " << v->get()
			//}

			//std::cout << g->name << " after checking valves" << std::endl;
			if (check == false) {
				auto producer_set = g->producers->getSet();
				Signal sig_terminate("Terminate");
				for (auto & producer : producer_set) {
					//			std::cout << g->name << " Sending " << sig_terminate.msg() << std::endl;
					//123///Guard::putSignal(producer, sig_terminate);
					producer->putSignal(producer, sig_terminate);

				}
				//g->runTask();
				//123//Guard::putSignal(g, "Start");
				g->putSignal(g, "Start");
				GlobalCPUProfiler.endThread();
				return;
			}
			//////std::cout << "Check not Pass" << std::endl;
		}
	}

	static void ActionAtCheckingAndStart(Guard* g) {
		GlobalCPUProfiler.addThread();
		while (1) {
			bool check = true;
			while (check) {
				check = false;
				for (auto& v : g->vs) {
					if (v->check() == false)
						check = true;
				}
			}

			if (check == false) {
				GuardScheduler::putGuard(g->gs, g, "Start");
				GlobalCPUProfiler.endThread();
				return;
			}
			//////std::cout << "Check not Pass" << std::endl;
		}
	}

	static void ActionAtWait(Guard* g) {
		//std::cout << g->name << " At begin of ActionAtWait!" << std::endl;

		GlobalCPUProfiler.addThread();
		bool requested = g->is_leaf;
		if (requested) {
			for (Data *input_data : g->task->input) {
				g->request_version[input_data] = g->parent_version[input_data] + 1;
			}
		}
		while (1) {
			// check all children has complete.
			bool check = true;
			if (!g->is_leaf) {
				for (Data *output_data : g->task->output) {
					std::string statename_ = output_data->consumer()->gstate->get_statename();
					if ((statename_ != "Complete") && (statename_ != "Terminate")) {
						check = false;
						break;
					}
				}
				// all children completed
				if (check) {
					g->putSignal(g, "Completed");
					GlobalCPUProfiler.endThread();
					return;
				}
			}

			// If (this->children->request_version > this->version)
			//     this->request_version = parent_version + 1
			// requested can be considered as state D
			if (!requested) {  // if this node has not requested from its parent.
				// check all of its children, whether any of then request data.
				for (Data *output_data : g->task->output) {
					if (output_data->consumer()->request_version[output_data] >
					   output_data->version()) {
					   	// some child requested data, request data from all parents.
						for (Data *input_data : g->task->input) {
							g->request_version[input_data] = g->parent_version[input_data] + 1;
						}
						requested = true;
						break;
					}
				}
			}

			// If (this->parent_version + 1<= this->parent->version)
			//     GOTO R
			check = true;
			for (Data *input_data : g->task->input) {
				std::string statename_ = input_data->producer()->gstate->get_statename();
				if ((statename_ != "Complete") && (statename_ != "Terminate") &&
				   (g->parent_version[input_data] + 1 > input_data->version())) {
					check = false;
					break;
				}
			}

			if (check) {
				g->putSignal(g, "Start");
				GlobalCPUProfiler.endThread();
				return;
			}
		}
	}

	static void ActionAtCheckingNoStarting(Guard* g) { //123//switch to ActionAtCheckingAndTerminate
		GlobalCPUProfiler.addThread();
		while (1) {
			bool check = true;
			while (check) {
				check = false;
				for (auto v : g->vs) {
					if (v->check() == false)
						check = true;
				}
			}

			if (check == false) {
				//pass the valves!!
				//123//Guard::putSignal(g, "Start");
				g->putSignal(g, "Start");
				GlobalCPUProfiler.endThread();
				return;
			}
			assert(0);
		}
	}

	static void ActionAtEndchecking(Guard* g) {
		GlobalCPUProfiler.addThread();

		SyncLogger::print(g->name, " At the begin of ActionAtEndchecking");
		bool check = false;

		if (g->is_root) {
			g->putSignal(g, "Completed");
			GlobalCPUProfiler.endThread();
			return;
		}

		if (g->is_leaf) {
			// first check end valves (maybe empty)
			for (auto v : g->endvs) {
				if (v->check() == false)
					check = true;
			}
			// TODO: check input data and children

			if (check == false) {
				//pass the valves!!
				//123//Guard::putSignal(g, "Start");
				g->putSignal(g, "Completed");
				GlobalCPUProfiler.endThread();
				return;
			}
		}

		// check if all input data used completed version
		if (!g->is_root) {
			check = true;
			for (Data *input_data : g->task->input) {
				// get the state of its parent.
				std::string statename_ = input_data->producer()->gstate->get_statename();
				// if the parent is not finished or didn't used the complete version, check fail.
				if (((statename_ != "Complete") && (statename_ != "Terminate"))) //||   ////fix this for CNN
					//(input_data->version() > g->parent_version[input_data])) 
				{
					check = false;

					SyncLogger::print(g->name, " can not finish because ", input_data->producer()->name, " is ", statename_);
					break;
				}
			}
			// all input data used completed version
			if (check) {
				g->putSignal(g, "Completed");
				GlobalCPUProfiler.endThread();
				return;
			}
		}

		// check if all children completeds
		check = true;
		if (g->is_leaf) {
			check = false;
		} else {
			for (Data *output_data : g->task->output) {
				// get the state of its children.
				std::string statename_ = output_data->consumer()->gstate->get_statename();
				// if the parent is not finished or didn't used the complete version, check fail.
				if ((statename_ != "Complete") && (statename_ != "Terminate")) {
					check = false;
					break;
				}
			}
		}
		// all children completed version
		if (check) {
			g->putSignal(g, "Completed");
			GlobalCPUProfiler.endThread();
			return;
		}

		if (check != true) {
			g->putSignal(g, "Wait");
			GlobalCPUProfiler.endThread();
			return;
		}
		assert(0);
		
	}

	// this is asynced call, main program will continue to execute.
	static void ActionAtStart(Guard* g) {
		GlobalCPUProfiler.addThread();
		SyncLogger::prefix = "Guard-" + std::to_string(g->guardId) + " ";
		//std::cout << g->name << "static void ActionAtStart" << std::endl;

		// end valve, get the parent version for each input.
		// At begin of R: this->parent_version = this -> parent ->version
		for (Data *input_data : g->task->input) {
			g->parent_version[input_data] = input_data->version();
		}

		g->runTask(); 
		//SyncLogger::print("TASKFINISHED", g->name, "Wall: ", (g->TaskTimerWT).stop(), "CPU:", (g->TaskTimerCT).stop());

		GlobalCPUProfiler.endThread();

	}

	static void ActionAtPausing(Guard* g) {
		if (g->worker) g->worker->suspend();
	}

};


//Singleton...
//template<typedef StatesType>
//void GuardStateTransit<StatesType>::handle(Guard* g, GuardState* gs, Signal sig) {
//	if (sig.msg().compare("Check") == 0) {
//		GuardStateActions::ActionAtChecking(g);
//		return GuardStateTransit<StateCheck>::Instance();
//	}
//	else if (sig.msg().compare("Check") == 0) {
//	
//	}
//}

void GuardStateToInit::handle(Guard* g, Signal sig) {
	//GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
	//GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	//std::cout << g->name << " GuardStateToInit::handle: " << sig.msg() << std::endl;
	cputimer.start();
	walltimer.start();
	(g->StateTimerWT).start();
	(g->StateTimerCT).start();
	g->tag_log(g->name+" enter State::Init"+get_time());
	if (sig.msg().compare("Check") == 0 && g->execmodel == 0) {
		g->gstate = GuardStateToCheck::Instance();

		g->tag_log(g->name+" left State::Init"+get_time());
		g->tag_log(g->name+" enter State::StartCheck"+get_time());
		GuardStateActions::ActionAtCheckingNoStarting(g);
	} 
	else if (sig.msg().compare("Check") == 0 && g->execmodel == 1) {
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingAndTerminate(g);
	}
	else if (sig.msg().compare("Start") == 0 && g->execmodel == 2) {
		g->gstate = GuardStateToStart::Instance();
		GuardStateActions::ActionAtStart(g);

	}
	else if (sig.msg().compare("Terminate") == 0 && g->execmodel == 2) {
		//g->gstate = GuardStateToInit::Instance();
		//GuardStateActions::ActionAtSpecTerminated(g);
		assert(0);
	}
	else if (sig.msg().compare("Completed") == 0 && g->execmodel == 2) {
		//g->gstate = GuardStateToInit::Instance();
		//GuardStateActions::ActionAtSpecTerminated(g);
		assert(0);
	}
	else if (sig.msg().compare("Terminate") == 0) {
		g->gstate = GuardStateToTerminate::Instance();
		GuardStateActions::ActionAtTerminated(g);
	}
	else {
		assert(0); // not valid!
	}
	GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
	GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	return;
}

void GuardStateToCheck::handle(Guard* g, Signal sig) {
	//std::cout << g->name << " GuardStateToCheck::handle: " << sig.msg() << std::endl;
	(g->StateTimerWT).start();
	(g->StateTimerCT).start();
	if (sig.msg().compare("Check") == 0 && g->execmodel == 0) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingNoStarting(g);
	}
	else if (sig.msg().compare("Check") == 0 && g->execmodel == 1) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingAndTerminate(g);
	}
	else if (sig.msg().compare("Start") == 0) {
		g->gstate = GuardStateToStart::Instance();
		g->tag_log(g->name+" left State::StartCheck"+get_time());
		g->tag_log(g->name+" enter State::Execution"+get_time());
		GuardStateActions::ActionAtStart(g);

	}
	else if (sig.msg().compare("Terminate") == 0 && g->execmodel == 2) {
		g->gstate = GuardStateToInit::Instance();
		GuardStateActions::ActionAtSpecTerminated(g);
	}
	else if (sig.msg().compare("Terminate") == 0) {
		//assert(0);
		g->gstate = GuardStateToTerminate::Instance();
		GuardStateActions::ActionAtTerminated(g);
	}
	else if (sig.msg().compare("Completed") == 0 && g->execmodel == 2) {
		(g->StateTimerWT).start();
		(g->StateTimerCT).start();
		g->gstate = GuardStateToComplete::Instance();
		GuardStateActions::ActionAtCompleted(g);
		GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
		GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	}
	else {
		assert(0); // not valid!
	}
	GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
	GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	return;
}

void GuardStateToStart::handle(Guard* g, Signal sig) {

	//std::cout << g->name << " GuardStateToStart::handle: " << sig.msg() << std::endl;
	if (sig.msg().compare("Check") == 0 && g->execmodel == 0) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingNoStarting(g);
	}
	else if (sig.msg().compare("Check") == 0 && g->execmodel == 1) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingAndTerminate(g);
	}
	else if (sig.msg().compare("Check") == 0 && g->execmodel == 2) {
		g->gstate = GuardStateToCheck::Instance();
		GuardStateActions::ActionAtCheckingAndStart(g);
	}
	else if (sig.msg().compare("Completed") == 0) {
		(g->StateTimerWT).start();
		(g->StateTimerCT).start();
		g->gstate = GuardStateToComplete::Instance();
		GuardStateActions::ActionAtCompleted(g);
		GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
		GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	}
	else if (sig.msg().compare("Terminate") == 0 && g->execmodel == 2) {
		g->gstate = GuardStateToInit::Instance();
		GuardStateActions::ActionAtSpecTerminated(g);
	}
	else if (sig.msg().compare("Terminate") == 0) {
		g->gstate = GuardStateToTerminate::Instance();
		g->tag_log(g->name+" left State::Execution"+get_time());
		g->tag_log(g->name+" enter State::Completed"+get_time());
		GuardStateActions::ActionAtTerminated(g);
	}
	else if (sig.msg().compare("Endcheck") == 0) {
		//(g->StateTimerWT).start();
		//(g->StateTimerCT).start();
		g->gstate = GuardStateToEndcheck::Instance();
		g->tag_log(g->name+" left State::Execution"+get_time());
		g->tag_log(g->name+" enter State::Endcheck"+get_time());
		GuardStateActions::ActionAtEndchecking(g);
		//GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
		//GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	}
	else {
		assert(0); // not valid!
	}

	return;
}

void GuardStateToEndcheck::handle(Guard* g, Signal sig) {
	//std::cout << g->name << " GuardStateToEndcheck::handle: " << sig.msg() << std::endl;
	(g->StateTimerWT).start();
	(g->StateTimerCT).start();
	if (sig.msg().compare("Check") == 0 && g->execmodel == 0) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingNoStarting(g);
	}
	else if (sig.msg().compare("Check") == 0 && g->execmodel == 1) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingAndTerminate(g);
	}
	else if (sig.msg().compare("Start") == 0) {
		g->gstate = GuardStateToStart::Instance();
		g->tag_log(g->name+" left State::Endcheck"+get_time());
		g->tag_log(g->name+" enter State::Execution"+get_time());
		GuardStateActions::ActionAtStart(g);

	}
	else if (sig.msg().compare("Wait") == 0) {
		g->gstate = GuardStateToWait::Instance();
		g->tag_log(g->name+" left State::Endcheck"+get_time());
		g->tag_log(g->name+" enter State::Wait"+get_time());
		GuardStateActions::ActionAtWait(g);

	}
	else if (sig.msg().compare("Completed") == 0) {
		(g->StateTimerWT).start();
		(g->StateTimerCT).start();
		g->gstate = GuardStateToComplete::Instance();
		g->tag_log(g->name+" left State::Endcheck"+get_time());
		g->tag_log(g->name+" enter State::Completed"+get_time());
		GuardStateActions::ActionAtCompleted(g);
		GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
		GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	}
	else if (sig.msg().compare("Terminate") == 0 && g->execmodel == 2) {
		g->gstate = GuardStateToInit::Instance();
		GuardStateActions::ActionAtSpecTerminated(g);
	}
	else if (sig.msg().compare("Terminate") == 0) {
		//assert(0);
		g->gstate = GuardStateToTerminate::Instance();
		GuardStateActions::ActionAtTerminated(g);
	}
	else {
		assert(0); // not valid!
	}
	GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
	GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	return;
}

void GuardStateToWait::handle(Guard* g, Signal sig) {
	//std::cout << g->name << " GuardStateToCheck::handle: " << sig.msg() << std::endl;
	(g->StateTimerWT).start();
	(g->StateTimerCT).start();
	if (sig.msg().compare("Check") == 0 && g->execmodel == 0) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingNoStarting(g);
	}
	else if (sig.msg().compare("Check") == 0 && g->execmodel == 1) {
		assert(0);
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingAndTerminate(g);
	}
	else if (sig.msg().compare("Start") == 0) {
		g->gstate = GuardStateToStart::Instance();
		g->tag_log(g->name+" left State::Wait"+get_time());
		g->tag_log(g->name+" enter State::Execution"+get_time());
		GuardStateActions::ActionAtStart(g);

	}
	else if (sig.msg().compare("Completed") == 0) {
		(g->StateTimerWT).start();
		(g->StateTimerCT).start();
		g->gstate = GuardStateToComplete::Instance();
		g->tag_log(g->name+" left State::Wait"+get_time());
		g->tag_log(g->name+" enter State::Completed"+get_time());
		GuardStateActions::ActionAtCompleted(g);
		GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
		GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	}
	else if (sig.msg().compare("Terminate") == 0 && g->execmodel == 2) {
		g->gstate = GuardStateToInit::Instance();
		GuardStateActions::ActionAtSpecTerminated(g);
	}
	else if (sig.msg().compare("Terminate") == 0) {
		//assert(0);
		g->gstate = GuardStateToTerminate::Instance();
		GuardStateActions::ActionAtTerminated(g);
	}
	else {
		assert(0); // not valid!
	}
	GlobalProfiler.addWallTimeStamp(statename, (g->StateTimerWT).stop());
	GlobalProfiler.addCPUTimeStamp(statename, (g->StateTimerCT).stop());
	return;
}

void GuardStateToComplete::handle(Guard* g, Signal sig) {
	//std::cout << g->name << " GuardStateToComplete::handle: " << sig.msg() << std::endl;
	cputimer.start();
	walltimer.start();
	if (sig.msg().compare("Check") == 0 && g->execmodel == 0) {
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingNoStarting(g);
	}
	else if (sig.msg().compare("Check") == 0 && g->execmodel == 1) {
		g->gstate = GuardStateToCheck::Instance();

		GuardStateActions::ActionAtCheckingAndTerminate(g);
	}
	else if (sig.msg().compare("Terminate") == 0 && g->execmodel == 2) {
		assert(0);
	}
	else if (sig.msg().compare("Start") == 0 && g->execmodel == 2) {
		g->gstate = GuardStateToStart::Instance();
		GuardStateActions::ActionAtStart(g);

	}
	else if (sig.msg().compare("Completed") == 0 && g->execmodel == 2) {
		//assert(0);
	}
	else {
		assert(0); // not valid!
	}
	GlobalProfiler.addWallTimeStamp(statename, walltimer.stop());
	GlobalProfiler.addCPUTimeStamp(statename, cputimer.stop());
	return;
}


void GuardStateToPause::handle(Guard* g, Signal sig) {
	//std::cout << g->name << " GuardStateToPause::handle: " << sig.msg() << std::endl;
	cputimer.start();
	walltimer.start();
	if (sig.msg().compare("Check") == 0) {
		GuardStateActions::ActionAtCheckingNoStarting(g);
		g->gstate = GuardStateToCheck::Instance();
	}
	else {
		assert(0); // not valid!
	}
	GlobalProfiler.addWallTimeStamp(statename, walltimer.stop());
	GlobalProfiler.addCPUTimeStamp(statename, cputimer.stop());
	return;
}

void GuardStateToTerminate::handle(Guard* g, Signal sig) {
	//std::cout << g->name << " GuardStateToTerminate::handle: " << sig.msg() << std::endl;
	cputimer.start();
	walltimer.start();
	if (sig.msg().compare("Check") == 0) {
		//From Complete to Check??
		//Restart??
		GuardStateActions::ActionAtCheckingNoStarting(g);
		g->gstate = GuardStateToCheck::Instance();
	}
	else {
		assert(0); // not valid!
	}
	GlobalProfiler.addWallTimeStamp(statename, walltimer.stop());
	GlobalProfiler.addCPUTimeStamp(statename, cputimer.stop());
	return;
}

void GuardStateToResume::handle(Guard* g, Signal sig) {
	//std::cout << g->name << " GuardStateToResume::handle: " << sig.msg() << std::endl;
	cputimer.start();
	walltimer.start();
	if (sig.msg().compare("Check") == 0) {
		GuardStateActions::ActionAtCheckingNoStarting(g);
		g->gstate = GuardStateToCheck::Instance();
		return;
	}
	else {
		assert(0); // not valid!
	}
}



