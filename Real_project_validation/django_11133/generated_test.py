from django.http import HttpResponse
from django.test import SimpleTestCase


class HttpResponseMemoryviewTest(SimpleTestCase):
    def test_memoryview_content_returns_bytes(self):
        response = HttpResponse(memoryview(b"My Content"))
        self.assertEqual(response.content, b"My Content")

    def test_memoryview_empty_bytes(self):
        response = HttpResponse(memoryview(b""))
        self.assertEqual(response.content, b"")

    def test_memoryview_content_multiple_access(self):
        response = HttpResponse(memoryview(b"Test Data"))
        first_access = response.content
        second_access = response.content
        self.assertEqual(first_access, second_access)
        self.assertEqual(first_access, b"Test Data")

    def test_memoryview_content_setter(self):
        response = HttpResponse()
        response.content = memoryview(b"Updated")
        self.assertEqual(response.content, b"Updated")

    def test_memoryview_iteration(self):
        response = HttpResponse(memoryview(b"Hello World"))
        chunks = list(response)
        self.assertEqual(chunks, [b"Hello World"])
