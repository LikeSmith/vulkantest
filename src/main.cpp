/*
 * main.cpp
 * VulkanTest
 *
 * Author: Kyle Crandall
 * Date: DEC2023
 *
 * Description: test of basic vulkan usage
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <optional>
#include <set>
#include <string>
#include <algorithm>
#include <limits>
#include <fstream>

#include <cstdlib>
#include <cstdint>

// load additional vulkan functions
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* p_create_info, const VkAllocationCallbacks* p_allocator, VkDebugUtilsMessengerEXT* p_debug_messenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, p_create_info, p_allocator, p_debug_messenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger, const VkAllocationCallbacks* p_allocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, debug_messenger, p_allocator);
	}
}

class VulkanApp
{
public:
	void run()
	{
		init();
		mainloop();
		cleanup();
	}

private:
	// main flow functions
	void init()
	{
		// initialize window
		createWindow();

		// initialize vulkan
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createDevice();
		createSwapchain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createCommandBuffers();
		createSyncObjects();
	}

	void mainloop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(dev);
	}

	void cleanup()
	{
		for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(dev, image_available_semaphores[i], nullptr);
			vkDestroySemaphore(dev, render_finished_semaphores[i], nullptr);
			vkDestroyFence(dev, in_flight_fences[i], nullptr);
		}

		vkDestroyCommandPool(dev, command_pool, nullptr);
		vkDestroyPipeline(dev, graphics_pipeline, nullptr);
		vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
		vkDestroyRenderPass(dev, render_pass, nullptr);

		cleanupSwapchain();

		vkDestroyDevice(dev, nullptr);

		if (enable_validation_layers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
		}
		
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	// init process functions
	void createWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HIEGHT, "Vulkan App", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, window_resize_callback);
	}

	void createInstance()
	{
		if (enable_validation_layers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but are not available.");
		}

		VkApplicationInfo app_info{};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName = "Vulkan Test";
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName = "No Engine";
		app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo instance_create_info{};
		instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instance_create_info.pApplicationInfo = &app_info;

		std::vector<const char*> req_extensions = getRequiredExtensions();
		instance_create_info.enabledExtensionCount = static_cast<uint32_t>(req_extensions.size());
		instance_create_info.ppEnabledExtensionNames = req_extensions.data();

		if (enable_validation_layers)
		{
			instance_create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			instance_create_info.ppEnabledLayerNames = validation_layers.data();

			VkDebugUtilsMessengerCreateInfoEXT msgr_create_info{};
			msgr_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			msgr_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			msgr_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			msgr_create_info.pfnUserCallback = debugCallback;
			msgr_create_info.pUserData = nullptr;

			instance_create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&msgr_create_info;
		}
		else
		{
			instance_create_info.enabledLayerCount = 0;
			instance_create_info.pNext = nullptr;
		}

		if (vkCreateInstance(&instance_create_info, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create vulkan instance.");
		}
	}

	void setupDebugMessenger()
	{
		if (!enable_validation_layers) return;

		VkDebugUtilsMessengerCreateInfoEXT msgr_create_info{};
		msgr_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		msgr_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		msgr_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		msgr_create_info.pfnUserCallback = debugCallback;
		msgr_create_info.pUserData = nullptr;

		if (CreateDebugUtilsMessengerEXT(instance, &msgr_create_info, nullptr, &debug_messenger) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to setup debug messenger.");
		}
	}

	void pickPhysicalDevice()
	{
		uint32_t dev_count = 0;
		vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
		if (dev_count == 0)
		{
			throw std::runtime_error("failed to find GPUs with vulkan support.");
		}
		std::vector<VkPhysicalDevice> devs(dev_count);
		vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());

		for (const auto& dev : devs)
		{
			if (isDeviceSuitable(dev))
			{
				physical_device = dev;
				break;
			}
		}

		if (physical_device == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find suitable GPU");
		}
	}

	void createDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physical_device);
		float queue_priority = 1.0f;

		std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
		std::set<uint32_t> unique_queue_families = {
			indices.graphics_family.value(),
			indices.present_family.value()
		};
		VkPhysicalDeviceFeatures dev_feat{};
		VkDeviceCreateInfo dev_create_info{};

		for (uint32_t queue_fam : unique_queue_families)
		{
			VkDeviceQueueCreateInfo queue_create_info{};
			queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_create_info.queueFamilyIndex = queue_fam;
			queue_create_info.queueCount = 1;
			queue_create_info.pQueuePriorities = &queue_priority;
			queue_create_infos.push_back(queue_create_info);
		}

		dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		dev_create_info.pQueueCreateInfos = queue_create_infos.data();
		dev_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
		dev_create_info.pEnabledFeatures = &dev_feat;
		dev_create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
		dev_create_info.ppEnabledExtensionNames = device_extensions.data();

		if (enable_validation_layers)
		{
			dev_create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			dev_create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
			dev_create_info.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physical_device, &dev_create_info, nullptr, &dev) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device.");
		}

		vkGetDeviceQueue(dev, indices.graphics_family.value(), 0, &graphics_queue);
		vkGetDeviceQueue(dev, indices.graphics_family.value(), 0, &present_queue);
	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface.");
		}
	}

	void createSwapchain()
	{
		SwapChainSupportDetails details = quereySwapChainSupport(physical_device);
		QueueFamilyIndices indices = findQueueFamilies(physical_device);

		VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(details.formats);
		VkPresentModeKHR present_mode = chooseSwapPresentMode(details.present_modes);
		VkExtent2D extent = chooseSwapExtent(details.capabilities);
		uint32_t image_count = details.capabilities.minImageCount + 1;

		uint32_t queue_family_indices[] = { indices.graphics_family.value(), indices.present_family.value() };

		VkSwapchainCreateInfoKHR swap_create_info{};

		if (details.capabilities.maxImageCount > 0 && image_count > details.capabilities.maxImageCount)
		{
			image_count = details.capabilities.maxImageCount;
		}

		swap_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swap_create_info.surface = surface;
		swap_create_info.minImageCount = image_count;
		swap_create_info.imageFormat = surface_format.format;
		swap_create_info.imageColorSpace = surface_format.colorSpace;
		swap_create_info.imageExtent = extent;
		swap_create_info.imageArrayLayers = 1;
		swap_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		if (indices.graphics_family != indices.present_family)
		{
			swap_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			swap_create_info.queueFamilyIndexCount = 2;
			swap_create_info.pQueueFamilyIndices = queue_family_indices;
		}
		else
		{
			swap_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			swap_create_info.queueFamilyIndexCount = 0;
			swap_create_info.pQueueFamilyIndices = nullptr;
		}

		swap_create_info.preTransform = details.capabilities.currentTransform;
		swap_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		swap_create_info.presentMode = present_mode;
		swap_create_info.clipped = VK_TRUE;
		swap_create_info.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(dev, &swap_create_info, nullptr, &swapchain) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create swapchain.");
		}

		vkGetSwapchainImagesKHR(dev, swapchain, &image_count, nullptr);
		swapchain_images.resize(image_count);
		vkGetSwapchainImagesKHR(dev, swapchain, &image_count, swapchain_images.data());

		swapchain_image_format = surface_format.format;
		swapchain_extent = extent;
	}

	void createImageViews()
	{
		swapchain_image_views.resize(swapchain_images.size());

		for (size_t i = 0; i < swapchain_images.size(); i++)
		{
			VkImageViewCreateInfo view_create_info{};
			view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			view_create_info.image = swapchain_images[i];
			view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view_create_info.format = swapchain_image_format;
			view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			view_create_info.subresourceRange.baseMipLevel = 0;
			view_create_info.subresourceRange.levelCount = 1;
			view_create_info.subresourceRange.baseArrayLayer = 0;
			view_create_info.subresourceRange.layerCount = 1;

			if (vkCreateImageView(dev, &view_create_info, nullptr, &swapchain_image_views[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image views.");
			}
		}
	}

	void createRenderPass()
	{
		VkAttachmentDescription color_attachment{};
		VkAttachmentReference color_attachment_ref{};
		VkSubpassDescription subpass{};
		VkSubpassDependency subpass_dependency{};

		VkRenderPassCreateInfo render_pass_create_info{};

		color_attachment.format = swapchain_image_format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

		subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		subpass_dependency.dstSubpass = 0;
		subpass_dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpass_dependency.srcAccessMask = 0;
		subpass_dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_create_info.attachmentCount = 1;
		render_pass_create_info.pAttachments = &color_attachment;
		render_pass_create_info.subpassCount = 1;
		render_pass_create_info.pSubpasses = &subpass;
		render_pass_create_info.dependencyCount = 1;
		render_pass_create_info.pDependencies = &subpass_dependency;

		if (vkCreateRenderPass(dev, &render_pass_create_info, nullptr, &render_pass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass.");
		}
	}

	void createGraphicsPipeline()
	{
		std::vector<char> vert_shader_code = readFile("shaders/vert.spv");
		std::vector<char> frag_shader_code = readFile("shaders/frag.spv");

		VkShaderModule vert_shader = createShaderModule(vert_shader_code);
		VkShaderModule frag_shader = createShaderModule(frag_shader_code);

		std::vector<VkDynamicState> dynamic_states = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineColorBlendAttachmentState colorblend_attachment{};

		VkPipelineShaderStageCreateInfo vert_shader_create_info{};
		VkPipelineShaderStageCreateInfo frag_shader_create_info{};
		VkPipelineVertexInputStateCreateInfo vertex_create_info{};
		VkPipelineInputAssemblyStateCreateInfo assembly_create_info{};
		VkPipelineViewportStateCreateInfo viewport_create_info{};
		VkPipelineRasterizationStateCreateInfo rasterize_create_info{};
		VkPipelineMultisampleStateCreateInfo multisample_create_info{};
		VkPipelineColorBlendStateCreateInfo colorblend_create_info{};
		VkPipelineDynamicStateCreateInfo dynamic_state_create_info{};
		VkPipelineLayoutCreateInfo layout_create_info{};
		VkGraphicsPipelineCreateInfo pipeline_create_info{};

		vert_shader_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_shader_create_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vert_shader_create_info.module = vert_shader;
		vert_shader_create_info.pName = "main";

		frag_shader_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_shader_create_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_shader_create_info.module = frag_shader;
		frag_shader_create_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stages_create_info[] = { vert_shader_create_info, frag_shader_create_info };

		vertex_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_create_info.vertexBindingDescriptionCount = 0;
		vertex_create_info.pVertexBindingDescriptions = nullptr;
		vertex_create_info.vertexAttributeDescriptionCount = 0;
		vertex_create_info.pVertexAttributeDescriptions = nullptr;

		assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assembly_create_info.primitiveRestartEnable = VK_FALSE;

		viewport_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_create_info.viewportCount = 1;
		viewport_create_info.scissorCount = 1;

		rasterize_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterize_create_info.depthClampEnable = VK_FALSE;
		rasterize_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterize_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterize_create_info.lineWidth = 1.0f;
		rasterize_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterize_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterize_create_info.depthBiasEnable = VK_FALSE;
		rasterize_create_info.depthBiasConstantFactor = 0.0;
		rasterize_create_info.depthBiasClamp = 0.0f;
		rasterize_create_info.depthBiasSlopeFactor = 0.0f;

		multisample_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_create_info.sampleShadingEnable = VK_FALSE;
		multisample_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_create_info.minSampleShading = 1.0f;
		multisample_create_info.pSampleMask = nullptr;
		multisample_create_info.alphaToCoverageEnable = VK_FALSE;
		multisample_create_info.alphaToOneEnable = VK_FALSE;

		colorblend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorblend_attachment.blendEnable = VK_FALSE;
		colorblend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorblend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorblend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorblend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorblend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorblend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

		colorblend_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorblend_create_info.logicOpEnable = VK_FALSE;
		colorblend_create_info.logicOp = VK_LOGIC_OP_COPY;
		colorblend_create_info.attachmentCount = 1;
		colorblend_create_info.pAttachments = &colorblend_attachment;
		colorblend_create_info.blendConstants[0] = 0.0f;
		colorblend_create_info.blendConstants[1] = 0.0f;
		colorblend_create_info.blendConstants[2] = 0.0f;
		colorblend_create_info.blendConstants[3] = 0.0f;

		dynamic_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_state_create_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic_state_create_info.pDynamicStates = dynamic_states.data();

		layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layout_create_info.setLayoutCount = 0;
		layout_create_info.pSetLayouts = nullptr;
		layout_create_info.pushConstantRangeCount = 0;
		layout_create_info.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(dev, &layout_create_info, nullptr, &pipeline_layout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout.");
		}

		pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_create_info.stageCount = 2;
		pipeline_create_info.pStages = shader_stages_create_info;
		pipeline_create_info.pVertexInputState = &vertex_create_info;
		pipeline_create_info.pInputAssemblyState = &assembly_create_info;
		pipeline_create_info.pViewportState = &viewport_create_info;
		pipeline_create_info.pRasterizationState = &rasterize_create_info;
		pipeline_create_info.pMultisampleState = &multisample_create_info;
		pipeline_create_info.pDepthStencilState = nullptr;
		pipeline_create_info.pColorBlendState = &colorblend_create_info;
		pipeline_create_info.pDynamicState = &dynamic_state_create_info;
		pipeline_create_info.layout = pipeline_layout;
		pipeline_create_info.renderPass = render_pass;
		pipeline_create_info.subpass = 0;
		pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
		pipeline_create_info.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &graphics_pipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline.");
		}

		vkDestroyShaderModule(dev, vert_shader, nullptr);
		vkDestroyShaderModule(dev, frag_shader, nullptr);
	}

	void createFramebuffers()
	{
		swapchain_framebuffers.resize(swapchain_image_views.size());

		for (size_t i = 0; i < swapchain_image_views.size(); i++)
		{
			VkImageView attachments[] = {
				swapchain_image_views[i]
			};

			VkFramebufferCreateInfo frambuffer_create_info{};
			frambuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frambuffer_create_info.renderPass = render_pass;
			frambuffer_create_info.attachmentCount = 1;
			frambuffer_create_info.pAttachments = attachments;
			frambuffer_create_info.width = swapchain_extent.width;
			frambuffer_create_info.height = swapchain_extent.height;
			frambuffer_create_info.layers = 1;

			if (vkCreateFramebuffer(dev, &frambuffer_create_info, nullptr, &swapchain_framebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffers.");
			}
		}
	}

	void createCommandPool()
	{
		QueueFamilyIndices indices = findQueueFamilies(physical_device);

		VkCommandPoolCreateInfo cmd_pool_create_info{};

		cmd_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmd_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cmd_pool_create_info.queueFamilyIndex = indices.graphics_family.value();

		if (vkCreateCommandPool(dev, &cmd_pool_create_info, nullptr, &command_pool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool");
		}
	}

	void createCommandBuffers()
	{
		VkCommandBufferAllocateInfo cmd_buf_aloc_info{};

		command_buffers.resize(MAX_FRAMES_IN_FLIGHT);

		cmd_buf_aloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmd_buf_aloc_info.commandPool = command_pool;
		cmd_buf_aloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmd_buf_aloc_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

		if (vkAllocateCommandBuffers(dev, &cmd_buf_aloc_info, command_buffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed ot allocate command buffer.");
		}
	}

	void createSyncObjects()
	{
		VkSemaphoreCreateInfo semaphore_create_info{};
		VkFenceCreateInfo fence_create_info{};

		image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

		semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			if (
				vkCreateSemaphore(dev, &semaphore_create_info, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(dev, &semaphore_create_info, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
				vkCreateFence(dev, &fence_create_info, nullptr, &in_flight_fences[i])
				)
			{
				throw std::runtime_error("failed to create sync objects.");
			}
		}
	}

	// Main Loop functions
	void drawFrame()
	{
		uint32_t image_idx;
		VkSubmitInfo submit_info{};
		VkSemaphore wait_semaphores[] = { image_available_semaphores[cur_frame] };
		VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore signal_semaphores[] = { render_finished_semaphores[cur_frame] };
		VkPresentInfoKHR present_info{};
		VkSwapchainKHR swapchains[] = { swapchain };
		VkResult result;

		vkWaitForFences(dev, 1, &in_flight_fences[cur_frame], VK_TRUE, UINT64_MAX);
		
		result = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, image_available_semaphores[cur_frame], VK_NULL_HANDLE, &image_idx);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapchain();
			return;
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image.");
		}

		vkResetFences(dev, 1, &in_flight_fences[cur_frame]);

		vkResetCommandBuffer(command_buffers[cur_frame], 0);
		recordCommandBuffer(command_buffers[cur_frame], image_idx);

		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = wait_semaphores;
		submit_info.pWaitDstStageMask = wait_stages;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers[cur_frame];
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signal_semaphores;

		if (vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[cur_frame]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit command buffer.");
		}

		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swapchains;
		present_info.pImageIndices = &image_idx;
		present_info.pResults = nullptr;

		result = vkQueuePresentKHR(present_queue, &present_info);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window_resized)
		{
			window_resized = false;
			recreateSwapchain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image.");
		}

		cur_frame = (cur_frame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	// utility functions
	bool checkValidationLayerSupport()
	{
		uint32_t layer_count;
		vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

		std::vector<VkLayerProperties> available_layers(layer_count);
		vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());


		std::set<std::string> required_layers(validation_layers.begin(), validation_layers.end());
		for (const auto& layer : available_layers)
		{
			required_layers.erase(layer.layerName);
		}

		return required_layers.empty();
	}

	std::vector <const char*> getRequiredExtensions()
	{
		uint32_t glfw_extension_count = 0;
		const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
		std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

		if (enable_validation_layers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphics_family;
		std::optional<uint32_t> present_family;

		bool isComplete()
		{
			return
				graphics_family.has_value() &&
				present_family.has_value();
		}
	};

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice pdev)
	{
		QueueFamilyIndices indices;
		uint32_t queue_fam_count;
		vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_fam_count, nullptr);
		std::vector<VkQueueFamilyProperties> queue_families(queue_fam_count);
		vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_fam_count, queue_families.data());

		int i = 0;

		for (const auto& queue_fam : queue_families)
		{
			VkBool32 present_supported = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(pdev, i, surface, &present_supported);
			if (present_supported)
			{
				indices.present_family = i;
			}
			if (queue_fam.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphics_family = i;
			}

			if (indices.isComplete())
			{
				break;
			}

			i++;
		}

		return indices;
	}

	bool isDeviceSuitable(VkPhysicalDevice pdev)
	{
		QueueFamilyIndices indices = findQueueFamilies(pdev);
		bool ext_support = checkDeviceExtensionSupport(pdev);
		bool swap_chain_adequate = false;

		if (ext_support)
		{
			SwapChainSupportDetails details = quereySwapChainSupport(pdev);
			swap_chain_adequate = !details.formats.empty() && !details.present_modes.empty();
		}

		return indices.isComplete() && ext_support && swap_chain_adequate;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice pdev)
	{
		uint32_t ext_count;
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &ext_count, nullptr);
		std::vector<VkExtensionProperties> available_extensions(ext_count);
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &ext_count, available_extensions.data());

		std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());
		for (const auto ext : available_extensions)
		{
			required_extensions.erase(ext.extensionName);
		}

		return required_extensions.empty();
	}

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> present_modes;
	};

	SwapChainSupportDetails quereySwapChainSupport(VkPhysicalDevice pdev)
	{
		SwapChainSupportDetails details;
		uint32_t format_count;
		uint32_t mode_count;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pdev, surface, &details.capabilities);
		vkGetPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, nullptr);
		vkGetPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &mode_count, nullptr);
		
		if (format_count > 0)
		{
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, details.formats.data());
		}

		if (mode_count > 0)
		{
			details.present_modes.resize(mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &mode_count, details.present_modes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats)
	{
		for (const auto& format : formats)
		{
			if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return format;
			}
		}

		return formats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& modes)
	{
		for (const auto& mode : modes)
		{
			if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return mode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}

		int width, height;

		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}

	static std::vector<char> readFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file.");
		}

		size_t filesize = (size_t)file.tellg();
		std::vector<char> buffer(filesize);

		file.seekg(0);
		file.read(buffer.data(), filesize);
		file.close();

		return buffer;
	}

	VkShaderModule(createShaderModule(const std::vector<char>& code))
	{
		VkShaderModule shader;
		VkShaderModuleCreateInfo shader_create_info{};
		shader_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shader_create_info.codeSize = code.size();
		shader_create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

		if (vkCreateShaderModule(dev, &shader_create_info, nullptr, &shader) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module.");
		}

		return shader;
	}

	void recordCommandBuffer(VkCommandBuffer cmd_buf, uint32_t image_idx)
	{
		VkCommandBufferBeginInfo cmd_buf_begin_info{};
		VkRenderPassBeginInfo render_pass_begin_info{};
		VkClearValue clear_color = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
		VkViewport viewport{};
		VkRect2D scissor{};

		cmd_buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmd_buf_begin_info.flags = 0;
		cmd_buf_begin_info.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(cmd_buf, &cmd_buf_begin_info) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to begin command buffer.");
		}

		render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_begin_info.renderPass = render_pass;
		render_pass_begin_info.framebuffer = swapchain_framebuffers[image_idx];
		render_pass_begin_info.renderArea.offset = { 0, 0 };
		render_pass_begin_info.renderArea.extent = swapchain_extent;
		render_pass_begin_info.clearValueCount = 1;
		render_pass_begin_info.pClearValues = &clear_color;

		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapchain_extent.width);
		viewport.height = static_cast<float>(swapchain_extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		scissor.offset = { 0, 0 };
		scissor.extent = swapchain_extent;

		vkCmdBeginRenderPass(cmd_buf, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
		vkCmdSetViewport(cmd_buf, 0, 1, &viewport);
		vkCmdSetScissor(cmd_buf, 0, 1, &scissor);
		vkCmdDraw(cmd_buf, 3, 1, 0, 0);
		vkCmdEndRenderPass(cmd_buf);

		if (vkEndCommandBuffer(cmd_buf) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer.");
		}
	}

	void recreateSwapchain()
	{
		int width = 0;
		int height = 0;

		do 
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		} while (width == 0 || height == 0);

		vkDeviceWaitIdle(dev);

		cleanupSwapchain();

		createSwapchain();
		createImageViews();
		createFramebuffers();
	}

	void cleanupSwapchain()
	{
		for (auto framebuffer : swapchain_framebuffers)
		{
			vkDestroyFramebuffer(dev, framebuffer, nullptr);
		}
		for (auto img_view : swapchain_image_views)
		{
			vkDestroyImageView(dev, img_view, nullptr);
		}
		vkDestroySwapchainKHR(dev, swapchain, nullptr);
	}

	// callback functions
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data, void* p_user_date)
	{
		std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;
		return VK_FALSE;
	}

	static void window_resize_callback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window));
		app->window_resized = true;
	}

	// Window data
	const uint32_t WIDTH = 800;
	const uint32_t HIEGHT = 600;
	bool window_resized = false;
	GLFWwindow* window;

	// Vulkan Data
	const std::vector<const char*> validation_layers = {
		"VK_LAYER_KHRONOS_validation"
	};

	const std::vector<const char*> device_extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

#ifdef NDEBUG
	const bool enable_validation_layers = false;
#else
	const bool enable_validation_layers = true;
#endif

	// Vulkan interface data
	const int MAX_FRAMES_IN_FLIGHT = 2;
	uint32_t cur_frame = 0;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debug_messenger;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkDevice dev;
	VkQueue graphics_queue;
	VkQueue present_queue;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapchain;
	std::vector<VkImage> swapchain_images;
	std::vector<VkImageView> swapchain_image_views;
	std::vector<VkFramebuffer> swapchain_framebuffers;
	VkFormat swapchain_image_format;
	VkExtent2D swapchain_extent;
	VkRenderPass render_pass;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphics_pipeline;
	VkCommandPool command_pool;
	std::vector<VkCommandBuffer> command_buffers;
	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;
	std::vector<VkFence> in_flight_fences;
};

int main()
{
	VulkanApp app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
