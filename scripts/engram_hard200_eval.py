#!/usr/bin/env python3
"""Hard 200-question long-tail technical, API, protocol, and factual QA benchmark.

Completely disjoint from all Engram calibration corpora, 60-Q Tech QA, EVAL_FACTUAL (60/120),
ling_test_v2, and core_v14_local. Designed to be non-saturated on 4B-20B models so genuine
long-tail recall lifts are measurable with high signal-to-noise ratio.

Supports two evaluation modes:
  1. Live OpenAI-compatible server evaluation (`--base-url http://127.0.0.1:8080/v1`)
  2. Exporting a `prompt\\tanswer` TSV (`--export-tsv hard200.tsv`) for 4-way
     Base/Disabled/Real/Shuffled teacher-forced scoring via `rookery_engram_test --eval-factual`.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

# 200 exact-match long-tail items across 10 categories (20 per category).
# Each entry: (id, category, question, accepted_answers_list, completion_prompt_for_tsv, canonical_tsv_answer)
QUESTIONS: list[tuple[str, str, str, list[str], str, str]] = [
    # ------------------------------------------------------------------
    # 1. Linux syscalls, errno, POSIX kernel constants (20)
    # ------------------------------------------------------------------
    ("lx01", "linux_sys", "What is the numeric Linux x86_64 errno value for EAGAIN / EWOULDBLOCK? Reply with just the integer.", ["11"], "On Linux x86_64, the numeric errno value for EAGAIN is", " 11"),
    ("lx02", "linux_sys", "What is the numeric Linux x86_64 errno value for EACCES? Reply with just the integer.", ["13"], "On Linux x86_64, the numeric errno value for EACCES is", " 13"),
    ("lx03", "linux_sys", "What is the numeric Linux x86_64 errno value for EEXIST? Reply with just the integer.", ["17"], "On Linux x86_64, the numeric errno value for EEXIST is", " 17"),
    ("lx04", "linux_sys", "What is the numeric Linux x86_64 errno value for EINVAL? Reply with just the integer.", ["22"], "On Linux x86_64, the numeric errno value for EINVAL is", " 22"),
    ("lx05", "linux_sys", "What is the numeric Linux x86_64 errno value for EMFILE (too many open files)? Reply with just the integer.", ["24"], "On Linux x86_64, the numeric errno value for EMFILE is", " 24"),
    ("lx06", "linux_sys", "What is the numeric Linux x86_64 errno value for ENOSPC (no space left on device)? Reply with just the integer.", ["28"], "On Linux x86_64, the numeric errno value for ENOSPC is", " 28"),
    ("lx07", "linux_sys", "What is the numeric Linux x86_64 errno value for EPIPE (broken pipe)? Reply with just the integer.", ["32"], "On Linux x86_64, the numeric errno value for EPIPE is", " 32"),
    ("lx08", "linux_sys", "What is the numeric Linux x86_64 errno value for EDEADLK? Reply with just the integer.", ["35"], "On Linux x86_64, the numeric errno value for EDEADLK is", " 35"),
    ("lx09", "linux_sys", "What is the numeric Linux x86_64 errno value for ENOSYS (function not implemented)? Reply with just the integer.", ["38"], "On Linux x86_64, the numeric errno value for ENOSYS is", " 38"),
    ("lx10", "linux_sys", "What is the numeric Linux x86_64 errno value for ELOOP (too many levels of symbolic links)? Reply with just the integer.", ["40"], "On Linux x86_64, the numeric errno value for ELOOP is", " 40"),
    ("lx11", "linux_sys", "What is the numeric Linux x86_64 errno value for ECONNRESET? Reply with just the integer.", ["104"], "On Linux x86_64, the numeric errno value for ECONNRESET is", " 104"),
    ("lx12", "linux_sys", "What is the numeric Linux x86_64 errno value for ETIMEDOUT? Reply with just the integer.", ["110"], "On Linux x86_64, the numeric errno value for ETIMEDOUT is", " 110"),
    ("lx13", "linux_sys", "What is the numeric Linux x86_64 errno value for ECONNREFUSED? Reply with just the integer.", ["111"], "On Linux x86_64, the numeric errno value for ECONNREFUSED is", " 111"),
    ("lx14", "linux_sys", "What is the numeric Linux x86_64 errno value for EINPROGRESS? Reply with just the integer.", ["115"], "On Linux x86_64, the numeric errno value for EINPROGRESS is", " 115"),
    ("lx15", "linux_sys", "What is the Linux x86_64 syscall number for mmap? Reply with just the integer.", ["9"], "On Linux x86_64, the syscall number for mmap is", " 9"),
    ("lx16", "linux_sys", "What is the Linux x86_64 syscall number for ioctl? Reply with just the integer.", ["16"], "On Linux x86_64, the syscall number for ioctl is", " 16"),
    ("lx17", "linux_sys", "What is the Linux x86_64 syscall number for getpid? Reply with just the integer.", ["39"], "On Linux x86_64, the syscall number for getpid is", " 39"),
    ("lx18", "linux_sys", "What is the Linux x86_64 syscall number for fork? Reply with just the integer.", ["57"], "On Linux x86_64, the syscall number for fork is", " 57"),
    ("lx19", "linux_sys", "What is the Linux x86_64 syscall number for execve? Reply with just the integer.", ["59"], "On Linux x86_64, the syscall number for execve is", " 59"),
    ("lx20", "linux_sys", "What is the Linux x86_64 syscall number for exit_group? Reply with just the integer.", ["231"], "On Linux x86_64, the syscall number for exit_group is", " 231"),

    # ------------------------------------------------------------------
    # 2. C / C++ / POSIX / IEEE-754 / Binary Formats (20)
    # ------------------------------------------------------------------
    ("cp01", "c_cpp", "What is the IEEE-754 exponent bias for binary16 (FP16)? Reply with just the integer.", ["15"], "The IEEE-754 exponent bias for binary16 (FP16) is", " 15"),
    ("cp02", "c_cpp", "What is the IEEE-754 exponent bias for binary32 (FP32)? Reply with just the integer.", ["127"], "The IEEE-754 exponent bias for binary32 (FP32) is", " 127"),
    ("cp03", "c_cpp", "What is the IEEE-754 exponent bias for binary64 (FP64)? Reply with just the integer.", ["1023"], "The IEEE-754 exponent bias for binary64 (FP64) is", " 1023"),
    ("cp04", "c_cpp", "What is the exponent bias for FP8 E4M3 (OCP MX / E4M3FN)? Reply with just the integer.", ["7"], "The exponent bias for FP8 E4M3 is", " 7"),
    ("cp05", "c_cpp", "What is the exponent bias for FP8 E5M2? Reply with just the integer.", ["15"], "The exponent bias for FP8 E5M2 is", " 15"),
    ("cp06", "c_cpp", "What is the number of explicit significand (mantissa) bits in bfloat16 (BF16)? Reply with just the integer.", ["7"], "The number of explicit mantissa bits in bfloat16 is", " 7"),
    ("cp07", "c_cpp", "What is the size in bytes of an ELF64 file header (Elf64_Ehdr)? Reply with just the integer.", ["64"], "The size in bytes of an Elf64_Ehdr header is", " 64"),
    ("cp08", "c_cpp", "What is the size in bytes of an ELF64 program header entry (Elf64_Phdr)? Reply with just the integer.", ["56"], "The size in bytes of an Elf64_Phdr entry is", " 56"),
    ("cp09", "c_cpp", "What is the size in bytes of an ELF64 section header entry (Elf64_Shdr)? Reply with just the integer.", ["64"], "The size in bytes of an Elf64_Shdr entry is", " 64"),
    ("cp10", "c_cpp", "In POSIX C, which header declares dlopen and dlsym? Reply with just the header filename (e.g. stdio.h).", ["dlfcn.h", "<dlfcn.h>"], "In POSIX C, dlopen and dlsym are declared in header", " dlfcn.h"),
    ("cp11", "c_cpp", "In POSIX C, which header declares struct pollfd and poll()? Reply with just the header filename.", ["poll.h", "<poll.h>"], "In POSIX C, poll() is declared in header", " poll.h"),
    ("cp12", "c_cpp", "In Linux C, which header declares epoll_create1 and epoll_wait? Reply with just the header path.", ["sys/epoll.h", "<sys/epoll.h>"], "In Linux C, epoll_wait is declared in header", " sys/epoll.h"),
    ("cp13", "c_cpp", "In POSIX C, which header declares mmap and madvise? Reply with just the header path.", ["sys/mman.h", "<sys/mman.h>"], "In POSIX C, mmap and madvise are declared in header", " sys/mman.h"),
    ("cp14", "c_cpp", "In POSIX C, which header declares struct termios and tcsetattr? Reply with just the header filename.", ["termios.h", "<termios.h>"], "In POSIX C, tcsetattr is declared in header", " termios.h"),
    ("cp15", "c_cpp", "In C++20, which standard header defines std::bit_cast and std::popcount? Reply with just the header name.", ["bit", "<bit>"], "In C++20, std::bit_cast and std::popcount are defined in header", " <bit>"),
    ("cp16", "c_cpp", "In C++20, which standard header defines std::span? Reply with just the header name.", ["span", "<span>"], "In C++20, std::span is defined in header", " <span>"),
    ("cp17", "c_cpp", "In C++17, which standard header defines std::from_chars and std::to_chars? Reply with just the header name.", ["charconv", "<charconv>"], "In C++17, std::from_chars is defined in header", " <charconv>"),
    ("cp18", "c_cpp", "What is the exact value of Linux signal number SIGBUS on x86_64? Reply with just the integer.", ["7"], "On Linux x86_64, the signal number for SIGBUS is", " 7"),
    ("cp19", "c_cpp", "What is the exact value of Linux signal number SIGUSR1 on x86_64? Reply with just the integer.", ["10"], "On Linux x86_64, the signal number for SIGUSR1 is", " 10"),
    ("cp20", "c_cpp", "What is the exact value of Linux signal number SIGCHLD on x86_64? Reply with just the integer.", ["17"], "On Linux x86_64, the signal number for SIGCHLD is", " 17"),

    # ------------------------------------------------------------------
    # 3. Networking, RFCs, IP protocols, EtherTypes (20)
    # ------------------------------------------------------------------
    ("nt01", "net_rfc", "What is the RFC number that specifies TLS 1.3? Reply with just the integer.", ["8446"], "TLS 1.3 is specified in RFC", " 8446"),
    ("nt02", "net_rfc", "What is the RFC number that specifies QUIC transport protocol? Reply with just the integer.", ["9000"], "The QUIC transport protocol is specified in RFC", " 9000"),
    ("nt03", "net_rfc", "What is the RFC number that specifies HTTP/3? Reply with just the integer.", ["9114"], "HTTP/3 is specified in RFC", " 9114"),
    ("nt04", "net_rfc", "What is the RFC number that specifies the WebSocket protocol? Reply with just the integer.", ["6455"], "The WebSocket protocol is specified in RFC", " 6455"),
    ("nt05", "net_rfc", "What is the RFC number that specifies OAuth 2.0 Authorization Framework? Reply with just the integer.", ["6749"], "The OAuth 2.0 Authorization Framework is specified in RFC", " 6749"),
    ("nt06", "net_rfc", "What is the RFC number that specifies JSON Web Token (JWT)? Reply with just the integer.", ["7519"], "JSON Web Token (JWT) is specified in RFC", " 7519"),
    ("nt07", "net_rfc", "What is the RFC number that specifies the JSON data interchange format (current STD 90)? Reply with just the integer.", ["8259"], "The JSON data interchange format STD 90 is specified in RFC", " 8259"),
    ("nt08", "net_rfc", "What is the RFC number that specifies CBOR (Concise Binary Object Representation, STD 94)? Reply with just the integer.", ["8949"], "CBOR (STD 94) is specified in RFC", " 8949"),
    ("nt09", "net_rfc", "What is the RFC number that specifies BGP-4? Reply with just the integer.", ["4271"], "BGP-4 is specified in RFC", " 4271"),
    ("nt10", "net_rfc", "What is the RFC number that specifies IPv6 (STD 86, obsoleting 2460)? Reply with just the integer.", ["8200"], "IPv6 (STD 86) is specified in RFC", " 8200"),
    ("nt11", "net_rfc", "What is the RFC number that specifies UUID versions 1 through 8 (obsoleting RFC 4122)? Reply with just the integer.", ["9562"], "UUID versions 1 through 8 (obsoleting RFC 4122) are specified in RFC", " 9562"),
    ("nt12", "net_rfc", "What is the RFC number that specifies DNS Domain Names Implementation and Specification? Reply with just the integer.", ["1035"], "DNS implementation and specification is defined in RFC", " 1035"),
    ("nt13", "net_rfc", "What is the IP protocol number for GRE (Generic Routing Encapsulation)? Reply with just the integer.", ["47"], "The IP protocol number for GRE is", " 47"),
    ("nt14", "net_rfc", "What is the IP protocol number for IPsec ESP? Reply with just the integer.", ["50"], "The IP protocol number for IPsec ESP is", " 50"),
    ("nt15", "net_rfc", "What is the IP protocol number for ICMPv6? Reply with just the integer.", ["58"], "The IP protocol number for ICMPv6 is", " 58"),
    ("nt16", "net_rfc", "What is the IP protocol number for SCTP? Reply with just the integer.", ["132"], "The IP protocol number for SCTP is", " 132"),
    ("nt17", "net_rfc", "What is the 16-bit hexadecimal EtherType for IEEE 802.1Q VLAN tagging (with 0x prefix)? Reply with just the hex value.", ["0x8100", "8100"], "The 16-bit EtherType for IEEE 802.1Q VLAN tagging is", " 0x8100"),
    ("nt18", "net_rfc", "What is the 16-bit hexadecimal EtherType for IPv6 (with 0x prefix)? Reply with just the hex value.", ["0x86dd", "86dd"], "The 16-bit EtherType for IPv6 is", " 0x86DD"),
    ("nt19", "net_rfc", "What is the 16-bit hexadecimal EtherType for ARP (with 0x prefix)? Reply with just the hex value.", ["0x0806", "0806"], "The 16-bit EtherType for ARP is", " 0x0806"),
    ("nt20", "net_rfc", "What is the IANA default UDP port for RADIUS authentication? Reply with just the integer.", ["1812"], "The IANA default UDP port for RADIUS authentication is", " 1812"),

    # ------------------------------------------------------------------
    # 4. Python stdlib internals, struct codes, HTTP status codes (20)
    # ------------------------------------------------------------------
    ("py01", "python", "In Python's struct module, what single format character represents an unsigned 64-bit integer (unsigned long long)? Reply with just the character.", ["Q"], "In Python's struct module, the format character for unsigned 64-bit integer is", " Q"),
    ("py02", "python", "In Python's struct module, what single format character represents an IEEE-754 binary16 half-precision float? Reply with just the character.", ["e"], "In Python's struct module, the format character for IEEE-754 float16 is", " e"),
    ("py03", "python", "In Python's struct module, what single format character represents a signed 16-bit short integer? Reply with just the character.", ["h"], "In Python's struct module, the format character for signed 16-bit short is", " h"),
    ("py04", "python", "What HTTP status code is 'Unprocessable Content' (formerly Unprocessable Entity)? Reply with just the integer.", ["422"], "The HTTP status code for Unprocessable Content is", " 422"),
    ("py05", "python", "What HTTP status code is 'Precondition Required'? Reply with just the integer.", ["428"], "The HTTP status code for Precondition Required is", " 428"),
    ("py06", "python", "What HTTP status code is 'Unavailable For Legal Reasons'? Reply with just the integer.", ["451"], "The HTTP status code for Unavailable For Legal Reasons is", " 451"),
    ("py07", "python", "What HTTP status code is 'Insufficient Storage' (WebDAV)? Reply with just the integer.", ["507"], "The HTTP status code for Insufficient Storage is", " 507"),
    ("py08", "python", "What HTTP status code is 'Misdirected Request'? Reply with just the integer.", ["421"], "The HTTP status code for Misdirected Request is", " 421"),
    ("py09", "python", "In Python's standard library, which module provides the bisect_left function? Reply with just the module name.", ["bisect"], "In Python's standard library, bisect_left is in module", " bisect"),
    ("py10", "python", "In Python's standard library, which module provides the heappushpop function? Reply with just the module name.", ["heapq"], "In Python's standard library, heappushpop is in module", " heapq"),
    ("py11", "python", "In Python's standard library, which module provides ExitStack? Reply with just the module name.", ["contextlib"], "In Python's standard library, ExitStack is in module", " contextlib"),
    ("py12", "python", "In Python's standard library, which module provides singledispatch? Reply with just the module name.", ["functools"], "In Python's standard library, singledispatch is in module", " functools"),
    ("py13", "python", "In Python's standard library, which module provides the batched() iterator helper added in Python 3.12? Reply with just the module name.", ["itertools"], "In Python 3.12, batched() is in standard library module", " itertools"),
    ("py14", "python", "In Python's standard library, which module provides TopologicalSorter? Reply with just the module name.", ["graphlib"], "In Python's standard library, TopologicalSorter is in module", " graphlib"),
    ("py15", "python", "In Python's standard library, which module provides tomllib for parsing TOML? Reply with just the module name.", ["tomllib"], "In Python 3.11+, the built-in TOML parser module is named", " tomllib"),
    ("py16", "python", "In Python's standard library, which module provides the ZoneInfo class? Reply with just the module name.", ["zoneinfo"], "In Python's standard library, the ZoneInfo class is in module", " zoneinfo"),
    ("py17", "python", "In Python's standard library, what function in hmac compares two digests in constant time? Reply with just the function name.", ["compare_digest", "hmac.compare_digest"], "In Python's hmac module, the constant-time comparison function is", " compare_digest"),
    ("py18", "python", "In Python, what dunder method is called when an object is used as a key in a dict or member of a set to compute its hash? Reply with just the method name.", ["__hash__"], "In Python, the dunder method to compute an object's hash is", " __hash__"),
    ("py19", "python", "In Python, what dunder method is invoked by the 'with' statement upon exiting a context block? Reply with just the method name.", ["__exit__"], "In Python, the context manager dunder method called on block exit is", " __exit__"),
    ("py20", "python", "In Python's sys module, what function returns the default recursion limit? Reply with just the function name.", ["getrecursionlimit", "sys.getrecursionlimit"], "In Python's sys module, the function returning the recursion limit is", " getrecursionlimit"),

    # ------------------------------------------------------------------
    # 5. Git plumbing, magic bytes, Unix CLI internals (20)
    # ------------------------------------------------------------------
    ("gt01", "git_cli", "What Git plumbing command computes the object ID (and optionally writes a blob) from a file or stdin? Reply with just the subcommand (e.g. cat-file).", ["hash-object", "git hash-object"], "The Git plumbing subcommand that computes a blob object ID is git", " hash-object"),
    ("gt02", "git_cli", "What Git plumbing command creates a tree object from the current index? Reply with just the subcommand.", ["write-tree", "git write-tree"], "The Git plumbing subcommand that writes the index to a tree object is git", " write-tree"),
    ("gt03", "git_cli", "What Git plumbing command creates a new commit object directly from a tree SHA? Reply with just the subcommand.", ["commit-tree", "git commit-tree"], "The Git plumbing subcommand that creates a commit object from a tree is git", " commit-tree"),
    ("gt04", "git_cli", "What Git plumbing command safely updates the object name stored in a ref? Reply with just the subcommand.", ["update-ref", "git update-ref"], "The Git plumbing subcommand that safely updates a ref is git", " update-ref"),
    ("gt05", "git_cli", "What Git plumbing command reads or modifies symbolic refs such as HEAD? Reply with just the subcommand.", ["symbolic-ref", "git symbolic-ref"], "The Git plumbing subcommand that reads or modifies HEAD is git", " symbolic-ref"),
    ("gt06", "git_cli", "What Git plumbing command finds the best common ancestor(s) between two commits? Reply with just the subcommand.", ["merge-base", "git merge-base"], "The Git plumbing subcommand that finds the common ancestor of two commits is git", " merge-base"),
    ("gt07", "git_cli", "What Git plumbing command verifies the connectivity and validity of objects in the object database? Reply with just the subcommand.", ["fsck", "git fsck"], "The Git plumbing subcommand that verifies object database integrity is git", " fsck"),
    ("gt08", "git_cli", "What are the first 2 magic bytes of a gzip file in uppercase hex (e.g. 1F8B)? Reply with just the 4 hex digits.", ["1f8b", "0x1f8b", "1f 8b"], "The first two magic bytes of a gzip file in hex are", " 1F8B"),
    ("gt09", "git_cli", "What are the 4 magic bytes of a Zstandard (zstd) frame in little-endian hex (0xFD2FB528 in memory order: 28B52FFD)? Reply with just the 8 hex digits in file byte order.", ["28b52ffd", "28 b5 2f fd"], "In file byte order, the 4 magic bytes of a zstd frame are", " 28B52FFD"),
    ("gt10", "git_cli", "What are the 4 magic bytes of an LZ4 frame in file byte order (hex)? Reply with just the 8 hex digits.", ["04224d18", "04 22 4d 18"], "In file byte order, the 4 magic bytes of an LZ4 frame are", " 04224D18"),
    ("gt11", "git_cli", "What are the first 4 magic bytes of a PNG file in hex (file byte order)? Reply with just the 8 hex digits.", ["89504e47", "89 50 4e 47"], "The first 4 magic bytes of a PNG file in hex are", " 89504E47"),
    ("gt12", "git_cli", "What are the first 4 magic bytes of an ELF binary in hex (file byte order)? Reply with just the 8 hex digits.", ["7f454c46", "7f 45 4c 46"], "The first 4 magic bytes of an ELF binary in hex are", " 7F454C46"),
    ("gt13", "git_cli", "What are the first 4 magic bytes of a WebAssembly (.wasm) binary in hex (file byte order)? Reply with just the 8 hex digits.", ["0061736d", "00 61 73 6d"], "The first 4 magic bytes of a WebAssembly binary in hex are", " 0061736D"),
    ("gt14", "git_cli", "What 4-byte ASCII magic header starts every GGUF file? Reply with just the 4 characters.", ["GGUF"], "The 4-byte ASCII magic header of a GGUF file is", " GGUF"),
    ("gt15", "git_cli", "What exit code does GNU diff return when two files differ (and no error occurred)? Reply with just the integer.", ["1"], "GNU diff returns exit code", " 1"),
    ("gt16", "git_cli", "What exit code does GNU grep return when no lines matched (and no error occurred)? Reply with just the integer.", ["1"], "GNU grep returns exit code when no lines match:", " 1"),
    ("gt17", "git_cli", "In POSIX shell, what exit code indicates a command was found but is not executable? Reply with just the integer.", ["126"], "In POSIX shell, the exit code for command found but not executable is", " 126"),
    ("gt18", "git_cli", "In POSIX shell, what exit code indicates a command was not found? Reply with just the integer.", ["127"], "In POSIX shell, the exit code for command not found is", " 127"),
    ("gt19", "git_cli", "In GNU xargs, what short flag specifies the maximum number of parallel processes? Reply with just the flag (e.g. -P).", ["-P", "P"], "In GNU xargs, the short flag for max parallel processes is", " -P"),
    ("gt20", "git_cli", "In Linux sysctl, what kernel parameter controls the maximum number of queued incoming TCP connections for a listening socket (somaxconn)? Reply with the dotted key.", ["net.core.somaxconn"], "The Linux sysctl key for maximum socket listen backlog is", " net.core.somaxconn"),

    # ------------------------------------------------------------------
    # 6. Databases, SQLite, PostgreSQL internals & SQLSTATE codes (20)
    # ------------------------------------------------------------------
    ("db01", "db_sql", "What PostgreSQL SQLSTATE error code indicates unique_violation? Reply with just the 5-character code.", ["23505"], "The PostgreSQL SQLSTATE code for unique_violation is", " 23505"),
    ("db02", "db_sql", "What PostgreSQL SQLSTATE error code indicates foreign_key_violation? Reply with just the 5-character code.", ["23503"], "The PostgreSQL SQLSTATE code for foreign_key_violation is", " 23503"),
    ("db03", "db_sql", "What PostgreSQL SQLSTATE error code indicates not_null_violation? Reply with just the 5-character code.", ["23502"], "The PostgreSQL SQLSTATE code for not_null_violation is", " 23502"),
    ("db04", "db_sql", "What PostgreSQL SQLSTATE error code indicates serialization_failure? Reply with just the 5-character code.", ["40001"], "The PostgreSQL SQLSTATE code for serialization_failure is", " 40001"),
    ("db05", "db_sql", "What PostgreSQL SQLSTATE error code indicates deadlock_detected? Reply with just the 5-character code.", ["40p01"], "The PostgreSQL SQLSTATE code for deadlock_detected is", " 40P01"),
    ("db06", "db_sql", "What PostgreSQL SQLSTATE error code indicates undefined_table? Reply with just the 5-character code.", ["42p01"], "The PostgreSQL SQLSTATE code for undefined_table is", " 42P01"),
    ("db07", "db_sql", "What PostgreSQL SQLSTATE error code indicates undefined_column? Reply with just the 5-character code.", ["42703"], "The PostgreSQL SQLSTATE code for undefined_column is", " 42703"),
    ("db08", "db_sql", "What PostgreSQL SQLSTATE error code indicates invalid_text_representation? Reply with just the 5-character code.", ["22p02"], "The PostgreSQL SQLSTATE code for invalid_text_representation is", " 22P02"),
    ("db09", "db_sql", "In PostgreSQL, what system catalog table stores tables, indexes, sequences, and views? Reply with just the table name.", ["pg_class"], "In PostgreSQL, the system catalog storing relations is", " pg_class"),
    ("db10", "db_sql", "In PostgreSQL, what system catalog table stores table columns? Reply with just the table name.", ["pg_attribute"], "In PostgreSQL, the system catalog storing columns is", " pg_attribute"),
    ("db11", "db_sql", "In PostgreSQL, what system view shows one row per active server process and current query? Reply with just the view name.", ["pg_stat_activity"], "In PostgreSQL, active server sessions and queries are shown in view", " pg_stat_activity"),
    ("db12", "db_sql", "In PostgreSQL, what system view shows locks currently held or awaited by active processes? Reply with just the view name.", ["pg_locks"], "In PostgreSQL, active locks are shown in system view", " pg_locks"),
    ("db13", "db_sql", "In PostgreSQL, what hidden system column stores the physical location (block number, tuple index) of a row version? Reply with just the column name.", ["ctid"], "In PostgreSQL, the hidden system column storing physical tuple location is", " ctid"),
    ("db14", "db_sql", "In PostgreSQL, what hidden system column stores the inserting transaction ID of a row version? Reply with just the column name.", ["xmin"], "In PostgreSQL, the hidden system column storing the inserting transaction ID is", " xmin"),
    ("db15", "db_sql", "What is the exact size in bytes of the SQLite database file header? Reply with just the integer.", ["100"], "The SQLite database file header is exactly this many bytes:", " 100"),
    ("db16", "db_sql", "In SQLite, what PRAGMA enables write-ahead logging (PRAGMA <name> = WAL)? Reply with just the pragma name.", ["journal_mode"], "In SQLite, WAL is enabled via PRAGMA", " journal_mode"),
    ("db17", "db_sql", "In SQLite, what PRAGMA enforces foreign key constraints (PRAGMA <name> = ON)? Reply with just the pragma name.", ["foreign_keys"], "In SQLite, foreign key enforcement is enabled via PRAGMA", " foreign_keys"),
    ("db18", "db_sql", "In SQLite, what PRAGMA sets the busy timeout in milliseconds? Reply with just the pragma name.", ["busy_timeout"], "In SQLite, the wait timeout for locked databases is set via PRAGMA", " busy_timeout"),
    ("db19", "db_sql", "In SQLite, what PRAGMA checks database integrity and B-tree consistency? Reply with just the pragma name.", ["integrity_check"], "In SQLite, full B-tree consistency is verified via PRAGMA", " integrity_check"),
    ("db20", "db_sql", "In SQLite, what is the default page size in bytes since SQLite 3.12.0? Reply with just the integer.", ["4096"], "Since SQLite 3.12.0, the default database page size in bytes is", " 4096"),

    # ------------------------------------------------------------------
    # 7. Cryptography, Hashing, & Encoding Specifications (20)
    # ------------------------------------------------------------------
    ("cr01", "crypto", "What is the internal block size in bytes of SHA-256? Reply with just the integer.", ["64"], "The internal block size in bytes of SHA-256 is", " 64"),
    ("cr02", "crypto", "What is the internal block size in bytes of SHA-512? Reply with just the integer.", ["128"], "The internal block size in bytes of SHA-512 is", " 128"),
    ("cr03", "crypto", "What is the internal block size in bytes of SHA3-256 (Keccak rate r)? Reply with just the integer.", ["136"], "The internal block size (rate) in bytes of SHA3-256 is", " 136"),
    ("cr04", "crypto", "What is the internal block size in bytes of SHA3-512 (Keccak rate r)? Reply with just the integer.", ["72"], "The internal block size (rate) in bytes of SHA3-512 is", " 72"),
    ("cr05", "crypto", "What is the standard nonce size in bytes for AES-GCM (NIST SP 800-38D recommended)? Reply with just the integer.", ["12"], "The NIST recommended nonce size in bytes for AES-GCM is", " 12"),
    ("cr06", "crypto", "What is the authentication tag size in bytes for Poly1305? Reply with just the integer.", ["16"], "The authentication tag size in bytes for Poly1305 is", " 16"),
    ("cr07", "crypto", "What is the IETF RFC 8439 nonce size in bytes for ChaCha20-Poly1305? Reply with just the integer.", ["12"], "In RFC 8439, the nonce size in bytes for ChaCha20-Poly1305 is", " 12"),
    ("cr08", "crypto", "What is the nonce size in bytes for XChaCha20-Poly1305? Reply with just the integer.", ["24"], "The nonce size in bytes for XChaCha20-Poly1305 is", " 24"),
    ("cr09", "crypto", "What is the signature size in bytes of Ed25519? Reply with just the integer.", ["64"], "An Ed25519 signature has a fixed size in bytes of", " 64"),
    ("cr10", "crypto", "What is the public key size in bytes of Ed25519? Reply with just the integer.", ["32"], "An Ed25519 public key has a fixed size in bytes of", " 32"),
    ("cr11", "crypto", "What is the prime modulus p of Curve25519? Reply in the form 2^255 - 19.", ["2^255 - 19", "2^255-19"], "The prime modulus of Curve25519 is", " 2^255 - 19"),
    ("cr12", "crypto", "What is the number of rounds in AES-256? Reply with just the integer.", ["14"], "The number of encryption rounds in AES-256 is", " 14"),
    ("cr13", "crypto", "What is the number of rounds in AES-128? Reply with just the integer.", ["10"], "The number of encryption rounds in AES-128 is", " 10"),
    ("cr14", "crypto", "What is the number of rounds in AES-192? Reply with just the integer.", ["12"], "The number of encryption rounds in AES-192 is", " 12"),
    ("cr15", "crypto", "What is the chunk size in bytes used by the BLAKE3 Merkle tree? Reply with just the integer.", ["1024"], "The Merkle tree chunk size in bytes in BLAKE3 is", " 1024"),
    ("cr16", "crypto", "What is the number of rounds in BLAKE3's compression function? Reply with just the integer.", ["7"], "The number of rounds in BLAKE3's compression function is", " 7"),
    ("cr17", "crypto", "What is the standard output size in bytes of HMAC-SHA256? Reply with just the integer.", ["32"], "The output digest size in bytes of HMAC-SHA256 is", " 32"),
    ("cr18", "crypto", "What is the 32-bit FNV-1a prime in hexadecimal (with 0x prefix)? Reply with just the hex value.", ["0x01000193", "0x1000193"], "The 32-bit FNV-1a prime in hexadecimal is", " 0x01000193"),
    ("cr19", "crypto", "What is the highest valid Unicode code point in hexadecimal (with U+ or 0x prefix)? Reply with just the code point.", ["u+10ffff", "0x10ffff", "10ffff"], "The highest valid Unicode scalar value is", " U+10FFFF"),
    ("cr20", "crypto", "What is the hexadecimal byte value of the UTF-8 Byte Order Mark's first byte (with 0x prefix)? Reply with just the hex byte.", ["0xef", "ef"], "The first byte of the UTF-8 BOM in hexadecimal is", " 0xEF"),

    # ------------------------------------------------------------------
    # 8. Hardware, CPU/GPU ISA, CUDA, ABI, Quantization Specs (20)
    # ------------------------------------------------------------------
    ("hw01", "hw_gpu", "In the x86_64 System V AMD64 ABI, what register passes the 4th integer argument? Reply with just the register name.", ["rcx", "%rcx"], "In the x86_64 System V ABI, the 4th integer argument register is", " rcx"),
    ("hw02", "hw_gpu", "In the x86_64 System V AMD64 ABI, what register passes the 5th integer argument? Reply with just the register name.", ["r8", "%r8"], "In the x86_64 System V ABI, the 5th integer argument register is", " r8"),
    ("hw03", "hw_gpu", "In the Linux x86_64 syscall ABI, what register replaces rcx for the 4th syscall argument? Reply with just the register name.", ["r10", "%r10"], "In the Linux x86_64 syscall ABI, the 4th argument register is", " r10"),
    ("hw04", "hw_gpu", "In the AArch64 Linux syscall ABI, what register holds the syscall number? Reply with just the register name.", ["x8"], "In the AArch64 Linux syscall ABI, the syscall number register is", " x8"),
    ("hw05", "hw_gpu", "What is the CUDA compute capability (as major.minor, e.g. 7.0) of the NVIDIA Tesla V100? Reply with just major.minor.", ["7.0"], "The CUDA compute capability of the NVIDIA Tesla V100 is", " 7.0"),
    ("hw06", "hw_gpu", "What is the CUDA compute capability (as major.minor) of the NVIDIA Tesla P100? Reply with just major.minor.", ["6.0"], "The CUDA compute capability of the NVIDIA Tesla P100 is", " 6.0"),
    ("hw07", "hw_gpu", "What is the CUDA compute capability (as major.minor) of the NVIDIA GTX 1080 Ti / GTX 1060? Reply with just major.minor.", ["6.1"], "The CUDA compute capability of the NVIDIA GTX 1080 Ti is", " 6.1"),
    ("hw08", "hw_gpu", "What is the CUDA compute capability (as major.minor) of the NVIDIA RTX 3090? Reply with just major.minor.", ["8.6"], "The CUDA compute capability of the NVIDIA RTX 3090 is", " 8.6"),
    ("hw09", "hw_gpu", "What is the CUDA compute capability (as major.minor) of the NVIDIA A100? Reply with just major.minor.", ["8.0"], "The CUDA compute capability of the NVIDIA A100 is", " 8.0"),
    ("hw10", "hw_gpu", "What is the CUDA compute capability (as major.minor) of the NVIDIA RTX 4090? Reply with just major.minor.", ["8.9"], "The CUDA compute capability of the NVIDIA RTX 4090 is", " 8.9"),
    ("hw11", "hw_gpu", "What is the CUDA compute capability (as major.minor) of the NVIDIA H100? Reply with just major.minor.", ["9.0"], "The CUDA compute capability of the NVIDIA H100 is", " 9.0"),
    ("hw12", "hw_gpu", "What is the CUDA compute capability (as major.minor) of the NVIDIA Tesla T4? Reply with just major.minor.", ["7.5"], "The CUDA compute capability of the NVIDIA Tesla T4 is", " 7.5"),
    ("hw13", "hw_gpu", "What is the number of threads in an AMD RDNA wave32 wavefront? Reply with just the integer.", ["32"], "An AMD RDNA wave32 wavefront contains this many threads:", " 32"),
    ("hw14", "hw_gpu", "What is the line encoding scheme (e.g. 128b/130b) used by PCIe Gen3, Gen4, and Gen5? Reply with just the ratio.", ["128b/130b"], "PCIe Gen3, Gen4, and Gen5 use this line encoding:", " 128b/130b"),
    ("hw15", "hw_gpu", "What is the raw signaling rate in GT/s per lane for PCIe 4.0? Reply with just the integer.", ["16"], "The raw transfer rate per lane for PCIe 4.0 in GT/s is", " 16"),
    ("hw16", "hw_gpu", "What is the raw signaling rate in GT/s per lane for PCIe 5.0? Reply with just the integer.", ["32"], "The raw transfer rate per lane for PCIe 5.0 in GT/s is", " 32"),
    ("hw17", "hw_gpu", "How many bytes is a single GGML Q8_0 quantization block (32 weights)? Reply with just the integer.", ["34"], "A single GGML Q8_0 block (32 weights) occupies this many bytes:", " 34"),
    ("hw18", "hw_gpu", "How many bytes is a single GGML Q4_0 quantization block (32 weights)? Reply with just the integer.", ["18"], "A single GGML Q4_0 block (32 weights) occupies this many bytes:", " 18"),
    ("hw19", "hw_gpu", "How many bytes is a single GGML Q4_K super-block (256 weights)? Reply with just the integer.", ["144"], "A single GGML Q4_K super-block (256 weights) occupies this many bytes:", " 144"),
    ("hw20", "hw_gpu", "How many bytes is a single GGML Q6_K super-block (256 weights)? Reply with just the integer.", ["210"], "A single GGML Q6_K super-block (256 weights) occupies this many bytes:", " 210"),

    # ------------------------------------------------------------------
    # 9. Long-tail Chemistry, Physics, Math & Astronomy Facts (20)
    # ------------------------------------------------------------------
    ("sc01", "science", "What is the atomic number of Tungsten (W)? Reply with just the integer.", ["74"], "The atomic number of Tungsten (W) is", " 74"),
    ("sc02", "science", "What is the atomic number of Osmium (Os)? Reply with just the integer.", ["76"], "The atomic number of Osmium (Os) is", " 76"),
    ("sc03", "science", "What is the atomic number of Iridium (Ir)? Reply with just the integer.", ["77"], "The atomic number of Iridium (Ir) is", " 77"),
    ("sc04", "science", "What is the atomic number of Bismuth (Bi)? Reply with just the integer.", ["83"], "The atomic number of Bismuth (Bi) is", " 83"),
    ("sc05", "science", "What is the atomic number of Radon (Rn)? Reply with just the integer.", ["86"], "The atomic number of Radon (Rn) is", " 86"),
    ("sc06", "science", "What is the atomic number of Thorium (Th)? Reply with just the integer.", ["90"], "The atomic number of Thorium (Th) is", " 90"),
    ("sc07", "science", "What is the atomic number of Neodymium (Nd)? Reply with just the integer.", ["60"], "The atomic number of Neodymium (Nd) is", " 60"),
    ("sc08", "science", "What is the atomic number of Gadolinium (Gd)? Reply with just the integer.", ["64"], "The atomic number of Gadolinium (Gd) is", " 64"),
    ("sc09", "science", "What is the atomic number of Molybdenum (Mo)? Reply with just the integer.", ["42"], "The atomic number of Molybdenum (Mo) is", " 42"),
    ("sc10", "science", "What is the atomic number of Antimony (Sb)? Reply with just the integer.", ["51"], "The atomic number of Antimony (Sb) is", " 51"),
    ("sc11", "science", "What is the chemical symbol for Antimony? Reply with just the symbol.", ["sb"], "The chemical symbol for Antimony is", " Sb"),
    ("sc12", "science", "What is the chemical symbol for Tungsten? Reply with just the symbol.", ["w"], "The chemical symbol for Tungsten is", " W"),
    ("sc13", "science", "What is the chemical symbol for Tin? Reply with just the symbol.", ["sn"], "The chemical symbol for Tin is", " Sn"),
    ("sc14", "science", "What is the chemical symbol for Ruthenium? Reply with just the symbol.", ["ru"], "The chemical symbol for Ruthenium is", " Ru"),
    ("sc15", "science", "What is the chemical symbol for Praseodymium? Reply with just the symbol.", ["pr"], "The chemical symbol for Praseodymium is", " Pr"),
    ("sc16", "science", "What is the 8th prime number? Reply with just the integer.", ["19"], "The 8th prime number is", " 19"),
    ("sc17", "science", "What is the 15th prime number? Reply with just the integer.", ["47"], "The 15th prime number is", " 47"),
    ("sc18", "science", "What is the 25th prime number? Reply with just the integer.", ["97"], "The 25th prime number is", " 97"),
    ("sc19", "science", "What is the exact value of the speed of light in vacuum in m/s? Reply with just the integer.", ["299792458"], "The exact speed of light in vacuum in m/s is", " 299792458"),
    ("sc20", "science", "What is the largest moon of Neptune? Reply with just the name.", ["triton"], "The largest moon of Neptune is", " Triton"),

    # ------------------------------------------------------------------
    # 10. Long-tail World Capitals, Geography & ISO Codes (20)
    # ------------------------------------------------------------------
    ("ge01", "geo_iso", "What is the capital city of Kazakhstan? Reply with just the city name.", ["astana"], "The capital city of Kazakhstan is", " Astana"),
    ("ge02", "geo_iso", "What is the capital city of Myanmar? Reply with just the city name.", ["naypyidaw", "nay pyi taw"], "The capital city of Myanmar is", " Naypyidaw"),
    ("ge03", "geo_iso", "What is the capital city of Bhutan? Reply with just the city name.", ["thimphu"], "The capital city of Bhutan is", " Thimphu"),
    ("ge04", "geo_iso", "What is the capital city of Suriname? Reply with just the city name.", ["paramaribo"], "The capital city of Suriname is", " Paramaribo"),
    ("ge05", "geo_iso", "What is the capital city of Burkina Faso? Reply with just the city name.", ["ouagadougou"], "The capital city of Burkina Faso is", " Ouagadougou"),
    ("ge06", "geo_iso", "What is the capital city of Kyrgyzstan? Reply with just the city name.", ["bishkek"], "The capital city of Kyrgyzstan is", " Bishkek"),
    ("ge07", "geo_iso", "What is the capital city of Tajikistan? Reply with just the city name.", ["dushanbe"], "The capital city of Tajikistan is", " Dushanbe"),
    ("ge08", "geo_iso", "What is the capital city of Eritrea? Reply with just the city name.", ["asmara"], "The capital city of Eritrea is", " Asmara"),
    ("ge09", "geo_iso", "What is the capital city of Malawi? Reply with just the city name.", ["lilongwe"], "The capital city of Malawi is", " Lilongwe"),
    ("ge10", "geo_iso", "What is the capital city of Palau? Reply with just the city name.", ["ngerulmud"], "The capital city of Palau is", " Ngerulmud"),
    ("ge11", "geo_iso", "What is the capital city of Belize? Reply with just the city name.", ["belmopan"], "The capital city of Belize is", " Belmopan"),
    ("ge12", "geo_iso", "What is the capital city of Burundi (political capital since 2019)? Reply with just the city name.", ["gitega"], "The political capital city of Burundi is", " Gitega"),
    ("ge13", "geo_iso", "What is the capital city of Mauritania? Reply with just the city name.", ["nouakchott"], "The capital city of Mauritania is", " Nouakchott"),
    ("ge14", "geo_iso", "What is the capital city of Turkmenistan? Reply with just the city name.", ["ashgabat"], "The capital city of Turkmenistan is", " Ashgabat"),
    ("ge15", "geo_iso", "What is the capital city of Moldova? Reply with just the city name.", ["chisinau", "chișinău"], "The capital city of Moldova is", " Chisinau"),
    ("ge16", "geo_iso", "What is the ISO 4217 3-letter currency code for the Swiss Franc? Reply with just the 3 letters.", ["chf"], "The ISO 4217 currency code for the Swiss Franc is", " CHF"),
    ("ge17", "geo_iso", "What is the ISO 4217 3-letter currency code for the Polish Zloty? Reply with just the 3 letters.", ["pln"], "The ISO 4217 currency code for the Polish Zloty is", " PLN"),
    ("ge18", "geo_iso", "What is the ISO 4217 3-letter currency code for the South African Rand? Reply with just the 3 letters.", ["zar"], "The ISO 4217 currency code for the South African Rand is", " ZAR"),
    ("ge19", "geo_iso", "What is the ISO 3166-1 alpha-2 country code for Switzerland? Reply with just the 2 letters.", ["ch"], "The ISO 3166-1 alpha-2 country code for Switzerland is", " CH"),
    ("ge20", "geo_iso", "What is the ISO 3166-1 alpha-2 country code for Estonia? Reply with just the 2 letters.", ["ee"], "The ISO 3166-1 alpha-2 country code for Estonia is", " EE"),
]

assert len(QUESTIONS) == 200, f"Expected 200 questions, got {len(QUESTIONS)}"


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    s = re.sub(r"^```[^\n]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    s = s.strip().strip("`\"'.,:;!*()[] ")
    return s


def matches(pred: str, gold_list: list[str]) -> bool:
    p = norm(pred)
    if not p:
        return False
    first_line = p.splitlines()[0].strip().strip("`\"'.,:;!*()[] ")
    for g in gold_list:
        gl = g.lower().strip()
        if p == gl or first_line == gl:
            return True
        if len(gl) >= 3 and re.search(rf"(?<![a-z0-9_]){re.escape(gl)}(?![a-z0-9_])", first_line):
            return True
        if gl.isdigit() and re.search(rf"(?<![0-9]){re.escape(gl)}(?![0-9])", first_line):
            return True
    return False


def ask(base_url: str, model: str, q: str, enable_thinking: bool, engram_scale: float | None, engram_gate_bias: float | None) -> str:
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer concisely with only the exact requested value."},
            {"role": "user", "content": q},
        ],
        "temperature": 0.0,
        "max_tokens": 512 if enable_thinking else 32,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if engram_scale is not None:
        payload["engram_scale"] = engram_scale
    if engram_gate_bias is not None:
        payload["engram_gate_bias"] = engram_gate_bias
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    msg = body["choices"][0]["message"]
    return (msg.get("content") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", help="e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="model")
    ap.add_argument("--think", action="store_true", help="Enable thinking mode (default: disabled for crisp direct recall)")
    ap.add_argument("--engram-scale", type=float, default=None)
    ap.add_argument("--engram-gate-bias", type=float, default=None)
    ap.add_argument("--export-tsv", help="Export completion TSV for 4-way rookery_engram_test --eval-factual")
    ap.add_argument("--out-json", help="Save per-item results JSON")
    args = ap.parse_args()

    if args.export_tsv:
        p = Path(args.export_tsv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for _, _, _, _, prompt, ans in QUESTIONS:
                f.write(f"{prompt}\t{ans}\n")
        print(f"[hard200] Exported {len(QUESTIONS)} items to {p}")
        if not args.base_url:
            return

    if not args.base_url:
        ap.error("--base-url is required unless --export-tsv is used alone")

    passed = 0
    by_cat: dict[str, list[int]] = {}
    rows = []
    for qid, cat, q, ans, _, _ in QUESTIONS:
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][1] += 1
        pred = ask(args.base_url, args.model, q, args.think, args.engram_scale, args.engram_gate_bias)
        ok = matches(pred, ans)
        if ok:
            passed += 1
            by_cat[cat][0] += 1
        rows.append({"id": qid, "cat": cat, "ok": ok, "pred": pred, "gold": ans})

    pct = 100.0 * passed / len(QUESTIONS)
    cat_summary = {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())}
    print(f"[hard200] {args.model}: {passed}/{len(QUESTIONS)} ({pct:.2f}%) {cat_summary}", flush=True)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps({"model": args.model, "passed": passed, "total": len(QUESTIONS), "pct": pct, "by_cat": cat_summary, "rows": rows}, indent=2)
        )


if __name__ == "__main__":
    main()
