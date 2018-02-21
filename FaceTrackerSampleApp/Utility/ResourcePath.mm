#include <string>
#include "ViewController.h"

std::string resourcePath(const std::string& fileName, const std::string& fileType) {
    NSString *resourceFilePath = [[NSBundle bundleForClass:ViewController.class]
                               pathForResource:[NSString stringWithUTF8String:fileName.c_str()]
                               ofType:[NSString stringWithUTF8String:fileType.c_str()]
                               ];

    return [resourceFilePath UTF8String];
}
