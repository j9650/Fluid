#include "logger.h"
//static class mtx that shares accross the binary...
std::mutex SyncLogger::mtx;