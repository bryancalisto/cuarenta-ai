package cua

import "github.com/lithammer/shortuuid/v4"

func getUid() string {
	return shortuuid.New()
}
